"""Numbered regression suites from past reviews.

Kept as one module because they are a historical series (Issue N, C-N,
H-N), each pinning a specific defect. Splitting them by subsystem would
scatter that provenance for no benefit.

Split out of the former tests/test_kernel_db.py.
"""

from __future__ import annotations
import hashlib
import logging
import shutil
import textwrap
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from spice_kernel_db import KernelDB, parse_metakernel
from spice_kernel_db.config import Config, load_config, save_config, show_config
from spice_kernel_db.hashing import (
    canonicalize_mission,
    classify_kernel,
    guess_mission,
    sha256_file,
)
from spice_kernel_db.parser import parse_metakernel_text, write_metakernel
from spice_kernel_db.remote import (
    Probe,
    ProbeResult,
    RemoteMetakernel,
    check_mk_availability,
    list_remote_metakernels,
    list_remote_missions,
    resolve_kernel_urls,
    SPICE_SERVERS,
    _DIR_ENTRY_RE,
    _VERSION_TAG_RE,
)

from tests.helpers import _ConWrapper


class TestMissionManagement:
    NASA_URL = "https://naif.jpl.nasa.gov/pub/naif/"
    ESA_URL = "https://spiftp.esac.esa.int/data/SPICE/"

    def test_add_and_get_mission(self, db):
        db.add_mission(
            "JUICE", self.ESA_URL,
            "https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/mk/",
            dedup=True,
        )
        m = db.get_mission("JUICE")
        assert m is not None
        assert m["name"] == "JUICE"
        assert m["server_url"] == self.ESA_URL
        assert m["server_label"] == "ESA"
        assert m["dedup"] is True
        assert "JUICE/kernels/mk/" in m["mk_dir_url"]

    def test_get_mission_case_insensitive(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://example.com/mk/", True)
        assert db.get_mission("juice") is not None
        assert db.get_mission("Juice") is not None

    def test_get_mission_not_found(self, db):
        assert db.get_mission("NONEXISTENT") is None

    def test_list_missions(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://example.com/JUICE/mk/", True)
        db.add_mission("MRO", self.NASA_URL, "https://example.com/MRO/mk/", False)

        missions = db.list_missions()
        assert len(missions) == 2

        by_name = {m["name"]: m for m in missions}
        assert by_name["JUICE"]["server_label"] == "ESA"
        assert by_name["JUICE"]["dedup"] is True
        assert by_name["MRO"]["server_label"] == "NASA"
        assert by_name["MRO"]["dedup"] is False

    def test_list_missions_empty(self, db):
        assert db.list_missions() == []

    def test_remove_mission(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://example.com/mk/", True)
        assert db.remove_mission("JUICE") is True
        assert db.get_mission("JUICE") is None

    def test_remove_mission_case_insensitive(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://example.com/mk/", True)
        assert db.remove_mission("juice") is True
        assert db.get_mission("JUICE") is None

    def test_remove_mission_prefix(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://example.com/mk/", True)
        assert db.remove_mission("jui") is True
        assert db.get_mission("JUICE") is None

    def test_remove_nonexistent_mission(self, db):
        assert db.remove_mission("NONEXISTENT") is False

    def test_add_mission_replaces_existing(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://old.com/mk/", True)
        db.add_mission("JUICE", self.NASA_URL, "https://new.com/mk/", False)

        m = db.get_mission("JUICE")
        assert m["server_url"] == self.NASA_URL
        assert m["dedup"] is False
        assert m["mk_dir_url"] == "https://new.com/mk/"

    def test_custom_server_label(self, db):
        db.add_mission(
            "CUSTOM", "https://my-server.example.com/spice/",
            "https://my-server.example.com/spice/CUSTOM/mk/", True,
        )
        m = db.get_mission("CUSTOM")
        assert m["server_label"] == "custom"


class TestIssue1PathTraversalPathValues:
    """Issue 1: PATH_VALUES like /../../../tmp/evil could escape download dir."""

    def test_path_traversal_in_path_values_raises(self, tmp_path):
        """PATH_VALUES that resolve outside download_dir must raise ValueError."""
        from spice_kernel_db.db import _validate_path_values
        download_dir = tmp_path / "kernels"
        download_dir.mkdir()
        # This resolves to /tmp/evil, outside download_dir
        with pytest.raises(ValueError, match="outside.*download"):
            _validate_path_values(
                ["/../../../tmp/evil"],
                mk_dir=download_dir / "JUICE" / "mk",
                download_dir=download_dir,
            )

    def test_safe_path_values_accepted(self, tmp_path):
        """PATH_VALUES that stay within download_dir should pass."""
        from spice_kernel_db.db import _validate_path_values
        download_dir = tmp_path / "kernels"
        mk_dir = download_dir / "JUICE" / "mk"
        mk_dir.mkdir(parents=True)
        # '..' from mk/ goes to JUICE/, still within download_dir
        _validate_path_values(
            [".."],
            mk_dir=mk_dir,
            download_dir=download_dir,
        )


class TestIssue2SizeOnlySkipCheck:
    """Issue 2: Files skipped when size matches but hash may differ."""

    def test_same_size_wrong_hash_not_skipped(self, tmp_path):
        """A file on disk with matching size but wrong hash should be re-downloaded."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Create a file on disk at the expected destination
        dest = tmp_path / "JUICE" / "lsk" / "naif0012.tls"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"WRONG CONTENT!!")  # same size as correct

        correct_content = b"CORRECT CONTENT"
        assert len(correct_content) == dest.stat().st_size

        # The DB has no record of this file, so resolve_kernel returns None.
        # The key behavior: when dest.is_file() and size matches remote,
        # it should also check hash. Since there's no DB hash to compare,
        # it should still download.
        # We test the _should_skip_download helper directly.
        from spice_kernel_db.db import _should_skip_download
        skip = _should_skip_download(
            dest=dest,
            remote_size=len(correct_content),
            db_hash="abcdef1234567890" * 4,  # known hash from DB
            force=False,
        )
        assert skip is False, "Should not skip when hash mismatches"
        db.close()

    def test_same_size_correct_hash_skipped(self, tmp_path):
        """A file on disk with matching size AND hash should be skipped."""
        from spice_kernel_db.db import _should_skip_download
        dest = tmp_path / "naif0012.tls"
        dest.write_bytes(b"CORRECT CONTENT")
        real_hash = sha256_file(dest)

        skip = _should_skip_download(
            dest=dest,
            remote_size=dest.stat().st_size,
            db_hash=real_hash,
            force=False,
        )
        assert skip is True, "Should skip when both size and hash match"


class TestIssue3SymlinkWithoutHashValidation:
    """Issue 3: Symlinks created without verifying resolved file's hash."""

    def test_link_existing_skips_hash_mismatch(self, tmp_path):
        """_link_existing_kernels should not create symlink when hash mismatches."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a kernel file
        kernel = tmp_path / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("CORRECT LSK CONTENT")
        h = db.register_file(kernel, mission="JUICE")

        # Now corrupt the file on disk (different content, same location)
        kernel.write_text("CORRUPTED CONTENT!!")

        # Try to link — should skip because hash of file on disk != DB hash
        download_dir = tmp_path / "download"
        download_dir.mkdir()
        n_linked = db._link_existing_kernels(
            indices=[0],
            filenames=["naif0012.tls"],
            relpaths=["lsk/naif0012.tls"],
            download_dir=download_dir,
            mission="JUICE",
        )
        assert n_linked == 0, "Should not link file with hash mismatch"
        db.close()


class TestIssue4PartialDownloadDetection:
    """Issue 4: Partial downloads not detected."""

    def test_partial_download_raises(self, tmp_path):
        """download_kernel should raise if bytes written < Content-Length."""
        from spice_kernel_db.remote import download_kernel

        dest = tmp_path / "test.bsp"

        # Mock a response that claims 1000 bytes but only delivers 500
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "1000"}
        mock_resp.read = MagicMock(side_effect=[b"x" * 500, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("spice_kernel_db.remote.urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(IOError, match="[Pp]artial|[Ii]ncomplete"):
                download_kernel("http://example.com/test.bsp", dest)

        # Partial file should be cleaned up
        assert not dest.exists(), "Partial file should be deleted"

    def test_interrupted_download_leaves_no_temp_file(self, tmp_path):
        """Ctrl+C mid-download must not strand a partial .tmp next to dest."""
        from spice_kernel_db.remote import download_kernel

        dest = tmp_path / "test.bsp"
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "1000"}
        mock_resp.read = MagicMock(side_effect=[b"x" * 500, KeyboardInterrupt])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("spice_kernel_db.remote.urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(KeyboardInterrupt):
                download_kernel("http://example.com/test.bsp", dest)

        assert list(tmp_path.iterdir()) == []

    def test_zero_byte_download_raises(self, tmp_path):
        """download_kernel should raise on zero-byte downloads."""
        from spice_kernel_db.remote import download_kernel

        dest = tmp_path / "test.bsp"

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "1000"}
        mock_resp.read = MagicMock(return_value=b"")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("spice_kernel_db.remote.urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(IOError):
                download_kernel("http://example.com/test.bsp", dest)

    def test_successful_download_no_error(self, tmp_path):
        """A complete download should succeed without error."""
        from spice_kernel_db.remote import download_kernel

        dest = tmp_path / "test.bsp"
        content = b"x" * 1000

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "1000"}
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("spice_kernel_db.remote.urllib.request.urlopen", return_value=mock_resp):
            result_dest, result_sha = download_kernel(
                "http://example.com/test.bsp", dest,
            )

        assert result_dest == dest
        assert dest.read_bytes() == content
        # C4: download_kernel now returns the streamed sha256 alongside
        # the path so callers don't need a second pass over the file.
        assert result_sha == hashlib.sha256(content).hexdigest()


class TestIssue5RaceConditionDBState:
    """Issue 5 / C5: DB state can change during download while lock released.

    After C5, detection raises ConcurrentModificationError instead of just
    logging — the previous behavior silently continued and produced wrong
    output.
    """

    def test_state_change_raises_concurrent_modification(self, tmp_path, caplog):
        """When state changed since the pre-snapshot, _check_state_changed
        must raise so the caller can abort the operation cleanly."""
        from spice_kernel_db import ConcurrentModificationError
        db = KernelDB(tmp_path / "test.duckdb")

        kernel = tmp_path / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK CONTENT")
        db.register_file(kernel, mission="JUICE")

        pre_snapshot = db._snapshot_kernel_hashes(["naif0012.tls"], "JUICE")

        kernel.write_text("UPDATED LSK CONTENT")
        db.register_file(kernel, mission="JUICE")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ConcurrentModificationError):
                db._check_state_changed(
                    pre_snapshot, ["naif0012.tls"], "JUICE",
                )
        # Warning is still emitted for forensics
        assert any("DB state changed" in r.message for r in caplog.records)
        db.close()

    def test_no_change_returns_false(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK")
        db.register_file(kernel, mission="JUICE")
        snap = db._snapshot_kernel_hashes(["naif0012.tls"], "JUICE")
        # No mutation between snapshot and check → returns False
        assert db._check_state_changed(snap, ["naif0012.tls"], "JUICE") is False
        db.close()


class TestIssue6SymlinkCreationErrorsSilent:
    """Issue 6: Symlink creation errors silently ignored, counter wrong."""

    def test_symlink_failure_not_counted(self, tmp_path):
        """If symlink_to() fails, n_linked should NOT be incremented."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a kernel file
        kernel = tmp_path / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("CORRECT LSK CONTENT")
        db.register_file(kernel, mission="JUICE")

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        # Patch symlink_to to raise OSError
        with patch.object(Path, "symlink_to", side_effect=OSError("Permission denied")):
            n_linked = db._link_existing_kernels(
                indices=[0],
                filenames=["naif0012.tls"],
                relpaths=["lsk/naif0012.tls"],
                download_dir=download_dir,
                mission="JUICE",
            )
        assert n_linked == 0, "Should not count failed symlink"
        db.close()

    def test_symlink_failure_logs_warning(self, tmp_path, caplog):
        """Symlink failure should log a warning."""
        db = KernelDB(tmp_path / "test.duckdb")

        kernel = tmp_path / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("CORRECT LSK CONTENT")
        db.register_file(kernel, mission="JUICE")

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        with patch.object(Path, "symlink_to", side_effect=OSError("Permission denied")):
            with caplog.at_level(logging.WARNING):
                db._link_existing_kernels(
                    indices=[0],
                    filenames=["naif0012.tls"],
                    relpaths=["lsk/naif0012.tls"],
                    download_dir=download_dir,
                    mission="JUICE",
                )
        assert any("symlink" in r.message.lower() or "Permission" in r.message
                    for r in caplog.records)
        db.close()


class TestIssue7PreRegistrationHashVerification:
    """Issue 7: register_file should optionally verify expected hash."""

    def test_expected_hash_mismatch_raises(self, tmp_path):
        """register_file with wrong expected_hash should raise ValueError."""
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("LSK CONTENT")

        with pytest.raises(ValueError, match="[Hh]ash mismatch"):
            db.register_file(
                kernel, mission="JUICE",
                expected_hash="0000000000000000" * 4,
            )
        db.close()

    def test_expected_hash_match_succeeds(self, tmp_path):
        """register_file with correct expected_hash should succeed."""
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("LSK CONTENT")
        correct_hash = sha256_file(kernel)

        h = db.register_file(
            kernel, mission="JUICE",
            expected_hash=correct_hash,
        )
        assert h == correct_hash
        db.close()

    def test_no_expected_hash_still_works(self, tmp_path):
        """register_file without expected_hash should work as before."""
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("LSK CONTENT")

        h = db.register_file(kernel, mission="JUICE")
        assert h == sha256_file(kernel)
        db.close()


class TestIssue8BroadExceptMasksErrors:
    """Issue 8: Broad except Exception masks real errors like PermissionError."""

    def test_permission_error_in_scan_reraised(self, tmp_path):
        """PermissionError during scan should be re-raised, not swallowed."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Create a file that will trigger PermissionError during registration
        kernel_dir = tmp_path / "JUICE" / "kernels" / "lsk"
        kernel_dir.mkdir(parents=True)
        kernel = kernel_dir / "naif0012.tls"
        kernel.write_text("LSK CONTENT")

        with patch.object(
            db, "register_file",
            side_effect=PermissionError("Permission denied"),
        ):
            with pytest.raises(PermissionError):
                db.scan_directory(kernel_dir)
        db.close()

    def test_os_error_in_scan_collected(self, tmp_path):
        """OSError during scan should be collected, not re-raised."""
        db = KernelDB(tmp_path / "test.duckdb")

        kernel_dir = tmp_path / "JUICE" / "kernels" / "lsk"
        kernel_dir.mkdir(parents=True)
        kernel = kernel_dir / "naif0012.tls"
        kernel.write_text("LSK CONTENT")

        with patch.object(
            db, "register_file",
            side_effect=OSError("Disk full"),
        ):
            # OSError should be caught and collected, not re-raised
            count, _ = db.scan_directory(kernel_dir)
        assert count == 0
        db.close()


class TestIssue9PathTraversalRewriteMetakernel:
    """Issue 9: Path traversal in rewrite_metakernel via ../."""

    def test_path_escape_in_relpath_skipped(self, tmp_path):
        """Kernel relpaths containing ../ that escape link_root should be skipped."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a kernel so resolve_kernel finds it
        kernel = tmp_path / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK CONTENT")
        db.register_file(kernel, mission="JUICE")

        # Create a metakernel with a path-traversal relpath
        mk = tmp_path / "evil.tm"
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '.' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = (
                '$KERNELS/../../etc/naif0012.tls'
              )
            \\begintext
        """))

        output = tmp_path / "output.tm"
        link_root = tmp_path / "links"

        _, warnings = db.rewrite_metakernel(
            mk, output, mission="JUICE", link_root=str(link_root),
        )
        # The path-traversal entry should generate a warning
        assert any("outside" in w.lower() or "escape" in w.lower() or "traversal" in w.lower()
                    for w in warnings)
        db.close()


class TestIssue10ThreadPoolSeverityDistinction:
    """Issue 10: Download failures should distinguish retriable from fatal."""

    def test_403_classified_as_fatal(self, tmp_path):
        """403 Forbidden should be classified as FATAL in warnings."""
        from spice_kernel_db.remote import download_kernels_parallel

        def mock_download(*args, **kwargs):
            raise urllib.error.HTTPError(
                "http://example.com/test.bsp", 403, "Forbidden", {}, None
            )

        tasks = [("http://example.com/test.bsp", tmp_path / "test.bsp", "test.bsp")]
        with patch("spice_kernel_db.remote.download_kernel", side_effect=mock_download):
            _, warnings = download_kernels_parallel(tasks)

        assert len(warnings) == 1
        assert "FATAL" in warnings[0] or "403" in warnings[0]

    def test_500_classified_as_retriable(self, tmp_path):
        """500 Server Error should be classified as RETRIABLE in warnings."""
        from spice_kernel_db.remote import download_kernels_parallel

        def mock_download(*args, **kwargs):
            raise urllib.error.HTTPError(
                "http://example.com/test.bsp", 500, "Internal Server Error", {}, None
            )

        tasks = [("http://example.com/test.bsp", tmp_path / "test.bsp", "test.bsp")]
        with patch("spice_kernel_db.remote.download_kernel", side_effect=mock_download):
            _, warnings = download_kernels_parallel(tasks)

        assert len(warnings) == 1
        assert "RETRIABLE" in warnings[0] or "retry" in warnings[0].lower()


class TestIssue11SameFilenameDifferentHash:
    """Issue 11: Re-registering same filename with different content should update hash."""

    def test_reregister_updates_hash(self, tmp_path):
        """Registering a file with same name but different content should update the hash."""
        db = KernelDB(tmp_path / "test.duckdb")

        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("ORIGINAL CONTENT")
        h1 = db.register_file(kernel, mission="JUICE")

        # Change content (simulates updated kernel)
        kernel.write_text("UPDATED CONTENT!")
        h2 = db.register_file(kernel, mission="JUICE")

        assert h1 != h2, "Hashes should differ for different content"

        # The kernels table should now have the new hash with this filename
        row = db.con.execute(
            "SELECT sha256 FROM kernels WHERE filename = ?",
            ["naif0012.tls"],
        ).fetchall()
        hashes = {r[0] for r in row}
        assert h2 in hashes, "New hash should be in kernels table"

        db.close()

    def test_reregister_logs_warning(self, tmp_path, caplog):
        """Hash change for same filename should log a warning."""
        db = KernelDB(tmp_path / "test.duckdb")

        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("ORIGINAL CONTENT")
        db.register_file(kernel, mission="JUICE")

        kernel.write_text("UPDATED CONTENT!")
        with caplog.at_level(logging.WARNING):
            db.register_file(kernel, mission="JUICE")

        assert any("naif0012.tls" in r.message for r in caplog.records)
        db.close()

    def test_old_location_preserved(self, tmp_path):
        """Old location entry should still exist after hash update."""
        db = KernelDB(tmp_path / "test.duckdb")

        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("ORIGINAL CONTENT")
        h1 = db.register_file(kernel, mission="JUICE")

        kernel.write_text("UPDATED CONTENT!")
        h2 = db.register_file(kernel, mission="JUICE")

        # Both hashes should have location entries
        locs_h1 = db.find_by_hash(h1)
        # The old location entry is overwritten because same abs_path
        # but the old hash record in kernels table should still exist
        old_kernel = db.con.execute(
            "SELECT filename FROM kernels WHERE sha256 = ?", [h1]
        ).fetchone()
        assert old_kernel is not None, "Old hash record should still exist in kernels table"
        db.close()


class TestC1PathTraversalGetMetakernel:
    """C1: hostile metakernel with traversing relpaths must be refused."""

    def test_get_refuses_traversal_in_kernels_to_load(self, tmp_path):
        from spice_kernel_db.remote import _fetch_metakernel
        db = KernelDB(tmp_path / "test.duckdb")

        evil_tm = textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = (
                '$KERNELS/../../../../tmp/pwn.bsp'
              )
            \\begintext
        """)

        with patch(
            "spice_kernel_db.db._fetch_metakernel",
            return_value=(evil_tm, "https://evil.example/evil.tm"),
        ):
            with pytest.raises(ValueError, match="path-traversing|unsafe"):
                db.get_metakernel(
                    "https://evil.example/evil.tm",
                    download_dir=tmp_path / "downloads",
                    mission="JUICE",
                    yes=True,
                )

        # No file should have been written outside the download tree
        assert not (Path("/tmp") / "pwn.bsp").exists() or \
            (Path("/tmp") / "pwn.bsp").stat().st_size == 0
        db.close()

    def test_link_existing_refuses_unsafe_relpath(self, tmp_path):
        """_link_existing_kernels must skip relpaths that escape mission_root."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a real kernel
        kernel = tmp_path / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK")
        db.register_file(kernel, mission="JUICE")

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        n = db._link_existing_kernels(
            indices=[0],
            filenames=["naif0012.tls"],
            relpaths=["../../../tmp/evil.tls"],   # escapes mission_root
            download_dir=download_dir,
            mission="JUICE",
        )
        assert n == 0
        # No symlink anywhere outside the mission dir
        assert not (Path("/tmp") / "evil.tls").exists() or \
            not (Path("/tmp") / "evil.tls").is_symlink()
        db.close()


class TestC2HashGateByResolvedPath:
    """C2: hash verification must use the resolved local path, not the
    caller-supplied filename. Otherwise the gate is bypassed whenever the
    canonical filename in `kernels` differs from what the metakernel asks
    for (the documented path-suffix-match case)."""

    def test_link_skips_when_corrupted_via_path_suffix_match(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a file under its canonical (long) name
        canonical = tmp_path / "JUICE" / "spk" / "jup365_19900101_20500101.bsp"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"identical content")
        h = db.register_file(canonical, mission="JUICE")

        # Manually add a SECOND location for the same hash with a different
        # filename — simulates "same content, also lives at this path with
        # the short name."
        alt = tmp_path / "JUICE" / "spk" / "jup365.bsp"
        alt.write_bytes(b"identical content")
        db.con.execute(
            "INSERT INTO locations VALUES (?, ?, ?, ?, current_timestamp)",
            [h, str(alt.resolve()), "JUICE", None],
        )

        # Now CORRUPT the alt file — content no longer matches its
        # registered hash. The pre-fix code would skip the hash check
        # because `WHERE filename='jup365.bsp'` returns NULL (canonical
        # is the long name). The fix looks up by resolved path.
        alt.write_bytes(b"corrupted bytes!!!")

        download_dir = tmp_path / "download"
        download_dir.mkdir()
        n = db._link_existing_kernels(
            indices=[0],
            filenames=["jup365.bsp"],
            relpaths=["spk/jup365.bsp"],
            download_dir=download_dir,
            mission="JUICE",
        )
        assert n == 0, (
            "C2: link must be refused when the resolved file's hash "
            "no longer matches its locations row"
        )
        db.close()

    def test_link_succeeds_when_canonical_differs_but_content_matches(
        self, tmp_path,
    ):
        """The legitimate path-suffix case: same content, different name.
        After C2, this still works as long as the hash matches."""
        db = KernelDB(tmp_path / "test.duckdb")
        canonical = tmp_path / "JUICE" / "spk" / "jup365_19900101_20500101.bsp"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"identical content")
        h = db.register_file(canonical, mission="JUICE")

        alt = tmp_path / "JUICE" / "spk" / "jup365.bsp"
        alt.write_bytes(b"identical content")
        db.con.execute(
            "INSERT INTO locations VALUES (?, ?, ?, ?, current_timestamp)",
            [h, str(alt.resolve()), "JUICE", None],
        )

        download_dir = tmp_path / "download"
        download_dir.mkdir()
        n = db._link_existing_kernels(
            indices=[0],
            filenames=["jup365.bsp"],
            relpaths=["spk/jup365.bsp"],
            download_dir=download_dir,
            mission="JUICE",
        )
        assert n == 1
        link = download_dir / "JUICE" / "spk" / "jup365.bsp"
        assert link.is_symlink()
        db.close()


class TestC3SupersededBy:
    """C3: re-registering the SAME path with new content must mark the
    old kernels row superseded; resolution then ignores it and is
    deterministic. Two DIFFERENT paths with the same filename are NOT
    supersessions — both remain active."""

    def test_same_path_content_change_marks_superseded(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "JUICE" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("v1 content")
        h1 = db.register_file(kernel, mission="JUICE")

        kernel.write_text("v2 content")
        h2 = db.register_file(kernel, mission="JUICE")
        assert h1 != h2

        # Old row's superseded_by must point to the new hash
        old = db.con.execute(
            "SELECT superseded_by FROM kernels WHERE sha256 = ?", [h1],
        ).fetchone()
        assert old is not None and old[0] == h2

        new = db.con.execute(
            "SELECT superseded_by FROM kernels WHERE sha256 = ?", [h2],
        ).fetchone()
        assert new is not None and new[0] is None
        db.close()

    def test_resolution_skips_superseded(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "JUICE" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("v1")
        db.register_file(kernel, mission="JUICE")
        kernel.write_text("v2")
        h2 = db.register_file(kernel, mission="JUICE")

        # find_by_filename must return only the active (v2) row
        hits = db.find_by_filename("naif0012.tls")
        assert len(hits) == 1
        assert hits[0]["sha256"] == h2
        db.close()

    def test_different_paths_same_name_NOT_superseded(self, tmp_path):
        """Two distinct files sharing a filename are not supersessions."""
        db = KernelDB(tmp_path / "test.duckdb")
        a = tmp_path / "tree_a" / "de440.bsp"
        b = tmp_path / "tree_b" / "de440.bsp"
        a.parent.mkdir(parents=True)
        b.parent.mkdir(parents=True)
        a.write_bytes(b"content A")
        b.write_bytes(b"content B")
        ha = db.register_file(a, mission="A")
        hb = db.register_file(b, mission="B")
        assert ha != hb

        # Both rows should be active (neither superseded)
        rows = db.con.execute(
            "SELECT sha256, superseded_by FROM kernels "
            "WHERE filename = 'de440.bsp'",
        ).fetchall()
        assert len(rows) == 2
        assert all(r[1] is None for r in rows)
        db.close()

    def test_old_location_at_same_path_is_cleaned_up(self, tmp_path):
        """When the path is re-registered with new content, the stale
        old-hash locations row at that path must be removed (it's not
        physically there anymore)."""
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "x" / "foo.bsp"
        kernel.parent.mkdir(parents=True)
        kernel.write_bytes(b"v1")
        h1 = db.register_file(kernel, mission="m")
        kernel.write_bytes(b"v2")
        db.register_file(kernel, mission="m")

        # locations should only have the new hash at this path
        rows = db.con.execute(
            "SELECT sha256 FROM locations WHERE abs_path = ?",
            [str(kernel.resolve())],
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] != h1
        db.close()


class TestC4DownloadHashStreaming:
    """C4: hash is streamed during download and authoritative.

    Closes the TOCTOU window between "download finished" and
    "register_file re-hashes the file" — and provides the hook for
    external/manifest checksum verification by accepting
    `expected_hash`.
    """

    def test_download_returns_streamed_sha256(self, tmp_path):
        from spice_kernel_db.remote import download_kernel

        content = b"hello world"
        expected = hashlib.sha256(content).hexdigest()

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(content))}
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        dest = tmp_path / "x.bsp"
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   return_value=mock_resp):
            _, sha = download_kernel("http://example.com/x.bsp", dest)
        assert sha == expected

    def test_download_rejects_mismatched_expected_hash(self, tmp_path):
        from spice_kernel_db.remote import download_kernel

        content = b"actual content"
        bogus = "0" * 64  # wrong hash on purpose

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(content))}
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        dest = tmp_path / "x.bsp"
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   return_value=mock_resp):
            with pytest.raises(IOError, match="[Hh]ash mismatch"):
                download_kernel(
                    "http://example.com/x.bsp", dest, expected_hash=bogus,
                )
        # Partial file should be cleaned up
        assert not dest.exists()

    def test_download_accepts_matching_expected_hash(self, tmp_path):
        from spice_kernel_db.remote import download_kernel

        content = b"content"
        good = hashlib.sha256(content).hexdigest()

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(content))}
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        dest = tmp_path / "x.bsp"
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   return_value=mock_resp):
            _, sha = download_kernel(
                "http://example.com/x.bsp", dest, expected_hash=good,
            )
        assert sha == good
        assert dest.read_bytes() == content

    def test_parallel_propagates_expected_hashes_to_per_url_check(
        self, tmp_path,
    ):
        from spice_kernel_db.remote import download_kernels_parallel

        content = b"abc"
        good = hashlib.sha256(content).hexdigest()
        bogus = "f" * 64

        calls = {}
        def fake_download(url, dest, *, expected_hash=None, **kwargs):
            calls[url] = expected_hash
            if expected_hash and expected_hash != good:
                raise IOError("Hash mismatch")
            dest.write_bytes(content)
            return dest, good

        url_good = "http://example.com/good.bsp"
        url_bad = "http://example.com/bad.bsp"
        tasks = [
            (url_good, tmp_path / "good.bsp", "good.bsp"),
            (url_bad, tmp_path / "bad.bsp", "bad.bsp"),
        ]

        with patch("spice_kernel_db.remote.download_kernel",
                   side_effect=fake_download):
            results, warnings = download_kernels_parallel(
                tasks,
                expected_hashes={url_good: good, url_bad: bogus},
            )
        # The hash mismatch task aborts; the matching one succeeds
        assert len(results) == 1
        assert results[0][1] == good
        assert any("bad.bsp" in w for w in warnings)
        # Per-URL expected_hash was forwarded
        assert calls[url_good] == good
        assert calls[url_bad] == bogus

    def test_parallel_returns_path_and_hash_tuples(self, tmp_path):
        from spice_kernel_db.remote import download_kernels_parallel

        content = b"xyz"
        sha = hashlib.sha256(content).hexdigest()
        def fake_download(url, dest, **kwargs):
            dest.write_bytes(content)
            return dest, sha

        tasks = [
            ("http://example.com/a.bsp", tmp_path / "a.bsp", "a.bsp"),
        ]
        with patch("spice_kernel_db.remote.download_kernel",
                   side_effect=fake_download):
            results, warnings = download_kernels_parallel(tasks)
        assert warnings == []
        assert len(results) == 1
        path, h = results[0]
        assert path == tmp_path / "a.bsp"
        assert h == sha


class TestC5ConcurrencyHandling:
    """C5: race detection must abort, not silently continue. reacquire()
    retries transient lock contention."""

    def test_reacquire_retries_on_lock_contention(self, tmp_path):
        """reacquire() should retry connect() when the first attempt
        raises (transient write-lock contention from another process)."""
        db = KernelDB(tmp_path / "test.duckdb")
        db.release()

        import duckdb as _duck
        real_connect = _duck.connect
        calls = {"n": 0}

        def flaky_connect(path, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _duck.IOException("simulated lock contention")
            return real_connect(path, **kwargs)

        with patch("spice_kernel_db.db.duckdb.connect", side_effect=flaky_connect):
            db.reacquire(initial_delay=0.001)
        assert calls["n"] >= 3
        # And the connection is healthy now
        assert db.con is not None
        db.close()

    def test_get_metakernel_aborts_on_concurrent_modification(self, tmp_path):
        """A `get` whose download window overlaps another writer's
        registration must raise instead of silently continuing."""
        from spice_kernel_db import ConcurrentModificationError

        db = KernelDB(tmp_path / "test.duckdb")
        download_dir = tmp_path / "downloads"

        # Pre-register the kernel under a known hash
        kernel = tmp_path / "preexisting" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("v1 content")
        db.register_file(kernel, mission="JUICE")

        # The remote .tm references that kernel
        tm_text = textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = ( '$KERNELS/lsk/naif0012.tls' )
            \\begintext
        """)

        # Simulate a concurrent writer that mutates the kernel record
        # in between query_remote_sizes and the post-reacquire check.
        original_query = None
        def evil_query(urls):
            # While the lock is released, "another process" mutates
            kernel.write_text("v2 different content")
            db.reacquire()
            db.register_file(kernel, mission="JUICE")
            db.release()
            return {u: 100 for u in urls}

        with patch(
            "spice_kernel_db.db._fetch_metakernel",
            return_value=(tm_text, "https://e.example/x.tm"),
        ), patch(
            "spice_kernel_db.db.query_remote_sizes", side_effect=evil_query,
        ):
            with pytest.raises(ConcurrentModificationError):
                db.get_metakernel(
                    "https://e.example/x.tm",
                    download_dir=download_dir,
                    mission="JUICE",
                    yes=True,
                )
        db.close()


class TestC6Transactions:
    """C6: multi-statement operations must roll back on failure, not
    leave the DB partially mutated."""

    def test_index_metakernel_rolls_back_on_insert_failure(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        mk = tmp_path / "x.tm"
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = ( '$KERNELS/lsk/a.tls' '$KERNELS/lsk/b.tls' )
            \\begintext
        """))
        db.index_metakernel(mk)
        mk_resolved = str(mk.resolve())
        before = db.con.execute(
            "SELECT COUNT(*) FROM metakernel_entries WHERE mk_path = ?",
            [mk_resolved],
        ).fetchone()[0]
        assert before == 2

        real = db.con
        db.con = _ConWrapper(real, "INSERT INTO METAKERNEL_ENTRIES", raise_after=2)
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                db.index_metakernel(mk)
        finally:
            db.con = real

        after = db.con.execute(
            "SELECT COUNT(*) FROM metakernel_entries WHERE mk_path = ?",
            [mk_resolved],
        ).fetchone()[0]
        assert after == 2, "C6: index_metakernel must roll back on failure"
        db.close()

    def test_register_file_rolls_back_on_failure(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "x.bsp"
        kernel.write_bytes(b"v1")
        db.register_file(kernel, mission="m")
        n_k = db.con.execute("SELECT COUNT(*) FROM kernels").fetchone()[0]
        n_l = db.con.execute("SELECT COUNT(*) FROM locations").fetchone()[0]

        kernel.write_bytes(b"v2 different content")
        real = db.con
        db.con = _ConWrapper(real, "INSERT OR REPLACE INTO LOCATIONS")
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                db.register_file(kernel, mission="m")
        finally:
            db.con = real

        assert db.con.execute(
            "SELECT COUNT(*) FROM kernels"
        ).fetchone()[0] == n_k
        assert db.con.execute(
            "SELECT COUNT(*) FROM locations"
        ).fetchone()[0] == n_l
        db.close()


class TestC7AtomicFileOps:
    """C7: file/symlink mutations are atomic — no concurrent reader sees
    a missing or truncated file."""

    def test_atomic_write_text_leaves_no_tmp_on_success(self, tmp_path):
        from spice_kernel_db.parser import _atomic_write_text
        p = tmp_path / "out.txt"
        _atomic_write_text(p, "hello")
        assert p.read_text() == "hello"
        # No stray .tmp.* files
        assert list(tmp_path.glob("*.tmp.*")) == []

    def test_atomic_write_text_cleans_up_tmp_on_error(self, tmp_path):
        from spice_kernel_db.parser import _atomic_write_text
        p = tmp_path / "out.txt"
        # Force os.replace to fail
        import os as _os
        with patch.object(_os, "replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                _atomic_write_text(p, "content")
        # No partial files left behind
        assert list(tmp_path.glob("*.tmp.*")) == []

    def test_atomic_write_text_overwrites_existing(self, tmp_path):
        from spice_kernel_db.parser import _atomic_write_text
        p = tmp_path / "out.txt"
        p.write_text("old")
        _atomic_write_text(p, "new")
        assert p.read_text() == "new"

    def test_atomic_symlink_creates_new(self, tmp_path):
        from spice_kernel_db.db import _atomic_symlink
        target = tmp_path / "real.bsp"
        target.write_bytes(b"data")
        link = tmp_path / "link.bsp"
        _atomic_symlink(target, link)
        assert link.is_symlink()
        assert link.resolve() == target.resolve()

    def test_atomic_symlink_replaces_existing(self, tmp_path):
        from spice_kernel_db.db import _atomic_symlink
        old_target = tmp_path / "old.bsp"
        old_target.write_bytes(b"old")
        new_target = tmp_path / "new.bsp"
        new_target.write_bytes(b"new")
        link = tmp_path / "link.bsp"
        link.symlink_to(old_target)
        _atomic_symlink(new_target, link)
        assert link.resolve() == new_target.resolve()

    def test_atomic_symlink_cleans_up_tmp_on_error(self, tmp_path):
        from spice_kernel_db.db import _atomic_symlink
        target = tmp_path / "real.bsp"
        target.write_bytes(b"data")
        link = tmp_path / "link.bsp"
        import os as _os
        with patch.object(_os, "replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                _atomic_symlink(target, link)
        # The tmp symlink must be cleaned up
        assert list(tmp_path.glob("*.tmp.*")) == []


class TestC8CaseInsensitiveFilenames:
    """C8: filename lookups are case-insensitive so APFS/HFS+/Windows
    (case-insensitive FS) and ext4 (case-sensitive) behave the same.
    Kernel identity is the sha256; name casing is metadata."""

    def test_find_by_filename_case_insensitive_canonical(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        k = tmp_path / "lsk" / "naif0012.tls"
        k.parent.mkdir(parents=True)
        k.write_text("LSK")
        db.register_file(k, mission="JUICE")

        # Query with various casings — all should find the kernel
        for q in ("naif0012.tls", "NAIF0012.TLS", "Naif0012.Tls"):
            hits = db.find_by_filename(q)
            assert len(hits) == 1, f"query {q!r} should match"
            assert hits[0]["abs_path"] == str(k.resolve())
        db.close()

    def test_find_by_filename_case_insensitive_path_suffix(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        # Register a file whose canonical name is e.g. CamelCase
        k = tmp_path / "spk" / "De440.bsp"
        k.parent.mkdir(parents=True)
        k.write_bytes(b"content")
        db.register_file(k, mission="JUICE")

        # find_by_filename with lowercase still finds it via path_suffix
        # or canonical match
        hits = db.find_by_filename("de440.bsp")
        assert len(hits) == 1
        db.close()

    def test_resolve_kernel_handles_case_mismatch(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        k = tmp_path / "lsk" / "naif0012.tls"
        k.parent.mkdir(parents=True)
        k.write_text("LSK")
        db.register_file(k, mission="JUICE")

        # Metakernel might ask for uppercase
        resolved, _ = db.resolve_kernel("NAIF0012.TLS", preferred_mission="JUICE")
        assert resolved == str(k.resolve())
        db.close()

    def test_register_file_detects_case_differing_collision(self, tmp_path):
        """If a file with the same name but different casing AND different
        content gets registered, the C8-aware detection should treat that
        as a same-filename collision (same path = supersession, different
        path = both active)."""
        db = KernelDB(tmp_path / "test.duckdb")

        # First file: lowercase
        a = tmp_path / "a" / "naif0012.tls"
        a.parent.mkdir(parents=True)
        a.write_text("v1")
        db.register_file(a, mission="m")

        # Second file: uppercase, different content, different path
        b = tmp_path / "b" / "NAIF0012.TLS"
        b.parent.mkdir(parents=True)
        b.write_text("v2")
        db.register_file(b, mission="m")

        # Both should be active (different paths, not supersession),
        # but the case-insensitive collision warning should have fired
        rows = db.con.execute(
            "SELECT filename, superseded_by FROM kernels "
            "ORDER BY filename",
        ).fetchall()
        assert len(rows) == 2
        assert all(r[1] is None for r in rows), \
            "different paths should not auto-supersede"
        db.close()


class TestH1DanglingSymlinkRepair:
    """H1: _link_existing_kernels must repair dangling symlinks instead
    of silently skipping them. The old code reported success while
    SPICE then failed at furnsh."""

    def test_dangling_symlink_is_repaired(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "JUICE" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK")
        db.register_file(kernel, mission="JUICE")

        download_dir = tmp_path / "download"
        link_target = download_dir / "JUICE" / "lsk" / "naif0012.tls"
        link_target.parent.mkdir(parents=True)

        # Create a dangling symlink (target doesn't exist)
        ghost = tmp_path / "nowhere" / "old.tls"
        link_target.symlink_to(ghost)
        assert link_target.is_symlink() and not link_target.exists()

        n = db._link_existing_kernels(
            indices=[0],
            filenames=["naif0012.tls"],
            relpaths=["lsk/naif0012.tls"],
            download_dir=download_dir,
            mission="JUICE",
        )
        assert n == 1
        # Now points at the real kernel
        assert link_target.resolve() == kernel.resolve()
        db.close()


class TestH2ParserRobustness:
    """H2: parser tolerates legal SPICE syntax that the old regex broke
    on — markers inside comments, ')' inside strings, '' escape."""

    def test_begindata_token_inside_comment_is_ignored(self, tmp_path):
        from spice_kernel_db.parser import parse_metakernel
        mk = tmp_path / "x.tm"
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begintext
            Inside a comment we discuss \\begindata syntax as an example.
            The parser must NOT treat the line above as a real marker.
            \\begintext
            \\begindata
              KERNELS_TO_LOAD = ( '$KERNELS/lsk/naif.tls' )
            \\begintext
        """))
        parsed = parse_metakernel(mk)
        assert parsed.kernels == ["$KERNELS/lsk/naif.tls"], (
            "the example token inside the comment block leaked into the "
            "data region under the old regex split"
        )

    def test_paren_inside_quoted_string_does_not_truncate_list(
        self, tmp_path,
    ):
        from spice_kernel_db.parser import parse_metakernel
        mk = tmp_path / "x.tm"
        # A kernel path containing ')' would have prematurely closed
        # the list under the old non-greedy regex.
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              KERNELS_TO_LOAD = (
                '/data/weird)name/foo.bsp'
                '/data/normal.bsp'
              )
            \\begintext
        """))
        parsed = parse_metakernel(mk)
        assert len(parsed.kernels) == 2
        assert parsed.kernels[0] == "/data/weird)name/foo.bsp"
        assert parsed.kernels[1] == "/data/normal.bsp"

    def test_double_quote_escape_in_string(self, tmp_path):
        from spice_kernel_db.parser import parse_metakernel
        mk = tmp_path / "x.tm"
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              KERNELS_TO_LOAD = ( '/data/it''s_quoted.bsp' )
            \\begintext
        """))
        parsed = parse_metakernel(mk)
        assert parsed.kernels == ["/data/it's_quoted.bsp"]

    def test_indented_begindata_token_works(self, tmp_path):
        """SPICE spec: marker at start of line (optionally indented)."""
        from spice_kernel_db.parser import parse_metakernel
        mk = tmp_path / "x.tm"
        mk.write_text(
            "KPL/MK\n"
            "   \\begindata\n"
            "  KERNELS_TO_LOAD = ( '$K/a.tls' )\n"
            "\\begintext\n"
        )
        parsed = parse_metakernel(mk)
        assert parsed.kernels == ["$K/a.tls"]


class TestH3PathSymbolPrefixCollision:
    """H3: longest PATH_SYMBOLS must be substituted first so an entry
    referencing ``$KERNELS_DATA`` isn't clobbered by an earlier
    ``$KERNELS`` replacement."""

    def test_resolve_prefers_longer_symbol(self):
        from spice_kernel_db.parser import parse_metakernel_text
        text = textwrap.dedent("""\
            \\begindata
              PATH_VALUES  = ( '/a' '/b' )
              PATH_SYMBOLS = ( 'KERNELS' 'KERNELS_DATA' )
              KERNELS_TO_LOAD = ( '$KERNELS_DATA/x.bsp' '$KERNELS/y.bsp' )
            \\begintext
        """)
        parsed = parse_metakernel_text(text, "test")
        # $KERNELS_DATA → /b, $KERNELS → /a — must NOT collapse the
        # longer symbol into /a_DATA via the str.replace prefix bug.
        assert parsed.resolve("$KERNELS_DATA/x.bsp") == "/b/x.bsp"
        assert parsed.resolve("$KERNELS/y.bsp") == "/a/y.bsp"

    def test_relpath_prefers_longer_symbol(self):
        from spice_kernel_db.parser import parse_metakernel_text
        text = textwrap.dedent("""\
            \\begindata
              PATH_VALUES  = ( '/a' '/b' )
              PATH_SYMBOLS = ( 'KERNELS' 'KERNELS_DATA' )
              KERNELS_TO_LOAD = ( '$KERNELS_DATA/x.bsp' '$KERNELS/y.bsp' )
            \\begintext
        """)
        parsed = parse_metakernel_text(text, "test")
        rels = parsed.kernel_relpaths()
        assert "x.bsp" in rels
        assert "y.bsp" in rels
        assert not any("_DATA" in r for r in rels), \
            "the longer symbol must be fully stripped"

    def test_resolve_kernel_urls_prefers_longer_symbol(self):
        from spice_kernel_db.parser import parse_metakernel_text
        from spice_kernel_db.remote import resolve_kernel_urls
        text = textwrap.dedent("""\
            \\begindata
              PATH_VALUES  = ( '..' 'extra' )
              PATH_SYMBOLS = ( 'KERNELS' 'KERNELS_DATA' )
              KERNELS_TO_LOAD = ( '$KERNELS_DATA/x.bsp' '$KERNELS/y.bsp' )
            \\begintext
        """)
        parsed = parse_metakernel_text(text, "test")
        urls = resolve_kernel_urls(
            "https://e.example/mission/mk/m.tm", parsed,
        )
        # $KERNELS_DATA must be resolved (and the longer one not clobbered)
        assert "_DATA" not in "".join(urls), urls
        assert any(u.endswith("x.bsp") for u in urls)
        assert any(u.endswith("y.bsp") for u in urls)


class TestH4MissionCanonicalisation:
    """H4: mission strings are canonicalised at every storage boundary
    so JUICE/juice/Juice converge to one row + one filesystem subtree."""

    def test_canonicalize_mission(self):
        assert canonicalize_mission("juice") == "JUICE"
        assert canonicalize_mission("JUICE") == "JUICE"
        assert canonicalize_mission("Juice") == "JUICE"
        # Sentinels stay lowercase
        assert canonicalize_mission("generic") == "generic"
        assert canonicalize_mission("Generic") == "generic"
        assert canonicalize_mission("UNKNOWN") == "unknown"
        # None / empty
        assert canonicalize_mission(None) == "unknown"
        assert canonicalize_mission("") == "unknown"
        # Whitespace
        assert canonicalize_mission("  juice  ") == "JUICE"

    def test_register_file_canonicalises_mission(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        k = tmp_path / "x.bsp"
        k.write_bytes(b"data")
        db.register_file(k, mission="juice")

        row = db.con.execute(
            "SELECT mission FROM locations"
        ).fetchone()
        assert row[0] == "JUICE"
        db.close()

    def test_add_mission_canonicalises(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        db.add_mission("juice", "http://x/", "http://x/mk/", dedup=True)
        rows = db.con.execute("SELECT name FROM missions").fetchall()
        assert rows == [("JUICE",)]
        db.close()

    def test_startup_warning_for_case_duplicates(self, tmp_path, caplog):
        """If a legacy DB has both `JUICE` and `juice` rows, opening it
        emits a warning so the user knows to merge."""
        # Manually insert duplicate-case rows BYPASSING canonicalisation
        db = KernelDB(tmp_path / "test.duckdb")
        db.con.execute(
            "INSERT INTO missions (name, server_url, mk_dir_url, dedup) "
            "VALUES ('JUICE', '', '', TRUE)"
        )
        db.con.execute(
            "INSERT INTO missions (name, server_url, mk_dir_url, dedup) "
            "VALUES ('juice', '', '', TRUE)"
        )
        db.close()

        with caplog.at_level(logging.WARNING):
            db2 = KernelDB(tmp_path / "test.duckdb")
        db2.close()
        assert any(
            "Mission case duplicates" in r.message
            for r in caplog.records
        )


class TestH5TaskInfoKeyedByPath:
    """H5: when two KERNELS_TO_LOAD entries share a basename under
    different relpaths, attribution must not get scrambled."""

    def test_duplicate_basename_in_metakernel_keeps_separate_attribution(
        self, tmp_path,
    ):
        """End-to-end: a hostile .tm with the same basename under two
        relpaths must result in both files being registered, each with
        its own source_url. This used to overwrite via task_info[name]."""
        db = KernelDB(tmp_path / "test.duckdb")
        download_dir = tmp_path / "downloads"

        tm_text = textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = (
                '$KERNELS/spk/de.bsp'
                '$KERNELS/spk_alt/de.bsp'
              )
            \\begintext
        """)

        # Mock the network so each URL yields distinct content
        def fake_dl(url, dest, *, expected_hash=None, **kwargs):
            content = b"A" * 10 if "spk/" in url else b"B" * 10
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            return dest, hashlib.sha256(content).hexdigest()

        with patch(
            "spice_kernel_db.db._fetch_metakernel",
            return_value=(tm_text, "https://e.example/m/mk/m.tm"),
        ), patch(
            "spice_kernel_db.db.query_remote_sizes",
            side_effect=lambda urls: {u: 10 for u in urls},
        ), patch(
            "spice_kernel_db.remote.download_kernel",
            side_effect=fake_dl,
        ):
            db.get_metakernel(
                "https://e.example/m/mk/m.tm",
                download_dir=download_dir,
                mission="JUICE",
                yes=True,
            )

        # Both files exist under their distinct relpaths with right content
        spk = download_dir / "JUICE" / "spk" / "de.bsp"
        spk_alt = download_dir / "JUICE" / "spk_alt" / "de.bsp"
        assert spk.read_bytes() == b"A" * 10
        assert spk_alt.read_bytes() == b"B" * 10

        # Each has its own location row with the correct source_url
        rows = db.con.execute(
            "SELECT abs_path, source_url FROM locations "
            "ORDER BY abs_path"
        ).fetchall()
        sources = {r[0]: r[1] for r in rows}
        assert "spk/de.bsp" in sources[str(spk.resolve())]
        assert "spk_alt/de.bsp" in sources[str(spk_alt.resolve())]
        db.close()


class TestH6StalenessDatetimeComparison:
    """H6: staleness check parses dates to datetime rather than doing
    lexicographic comparison on a 16-char timestamp prefix."""

    def test_subminute_acquire_detects_later_remote_correctly(
        self, tmp_path, capsys,
    ):
        from datetime import datetime, timedelta
        from spice_kernel_db.remote import RemoteMetakernel
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a fake metakernel with a known acquired_at
        mk_path = str(tmp_path / "x.tm")
        (tmp_path / "x.tm").write_text("\\begintext\n")
        acq = datetime(2025, 11, 27, 14, 30, 45)
        db.con.execute(
            "INSERT INTO metakernel_registry VALUES (?, ?, ?, ?, ?)",
            [mk_path, "JUICE", "https://e.example/mk/x.tm", "x.tm", acq],
        )

        # Remote modified 16 minutes later — clearly newer
        later = RemoteMetakernel(
            filename="x.tm",
            url="https://e.example/mk/x.tm",
            date="2025-11-27 14:46",
            size="1K",
            base_name="x.tm",
            version_tag=None,
        )
        with patch(
            "spice_kernel_db.db.list_remote_metakernels",
            return_value=[later],
        ):
            db._check_remote_staleness(mk_path, "JUICE")
        captured = capsys.readouterr().out
        assert "Remote update available" in captured
        db.close()

    def test_same_minute_acquire_is_NOT_flagged_stale(
        self, tmp_path, capsys,
    ):
        """If acquired_at is at 14:30:45 and the remote was modified at
        14:30 (same minute, before our acquisition), we must NOT warn."""
        from datetime import datetime
        from spice_kernel_db.remote import RemoteMetakernel
        db = KernelDB(tmp_path / "test.duckdb")

        mk_path = str(tmp_path / "x.tm")
        (tmp_path / "x.tm").write_text("\\begintext\n")
        acq = datetime(2025, 11, 27, 14, 30, 45)
        db.con.execute(
            "INSERT INTO metakernel_registry VALUES (?, ?, ?, ?, ?)",
            [mk_path, "JUICE", "https://e.example/mk/x.tm", "x.tm", acq],
        )
        same_minute = RemoteMetakernel(
            filename="x.tm",
            url="https://e.example/mk/x.tm",
            date="2025-11-27 14:30",
            size="1K",
            base_name="x.tm",
            version_tag=None,
        )
        with patch(
            "spice_kernel_db.db.list_remote_metakernels",
            return_value=[same_minute],
        ):
            db._check_remote_staleness(mk_path, "JUICE")
        captured = capsys.readouterr().out
        assert "Remote update available" not in captured
        db.close()


class TestUnsupersedePresentHashes:
    """Issue 29: a hash whose file is on disk must not stay superseded forever.

    `superseded_by` used to have exactly one writer, which only ever SET it.
    When anything put non-kernel content at a tracked path -- a pre-0.17.0
    interrupted download, a Dropbox conflict copy -- the real kernel was marked
    superseded by that junk. Restoring the file marked the junk superseded in
    turn, so BOTH rows ended up flagged, and find_by_filename() excludes
    superseded rows. resolve() then reported "not found" for a byte-perfect
    file and every `update` re-downloaded it, with no shipped repair path.

    The damaged state is built with SQL rather than by driving register_file,
    because the fix heals it on the first call: the point is that an existing
    database carrying this damage recovers, which is what a user's `scan` does.
    """

    def _damaged(self, db, path, content, junk_hash):
        path.write_text(content)
        good = sha256_file(path)
        db.con.execute(
            "INSERT INTO kernels (sha256, filename, kernel_type, size_bytes, "
            "superseded_by) VALUES (?, ?, ?, ?, ?)",
            [good, path.name, "pck", path.stat().st_size, junk_hash],
        )
        db.con.execute(
            "INSERT INTO locations VALUES (?, ?, ?, ?, current_timestamp)",
            [good, str(path), "JUICE", None],
        )
        return good

    def test_a_scan_heals_a_poisoned_row(self, db, tmp_path):
        p = tmp_path / "pck00011.tpc"
        junk = "f" * 64                      # a hash with no row and no location
        good = self._damaged(db, p, "REAL PCK CONTENT", junk)

        # the damage must be real, or this test proves nothing
        assert db.resolve_kernel("pck00011.tpc")[0] is None

        db.register_file(p, mission="JUICE")      # what `scan` does

        row = db.con.execute(
            "SELECT superseded_by FROM kernels WHERE sha256 = ?", [good]
        ).fetchone()
        assert row[0] is None, "flag not cleared for content present on disk"
        assert db.resolve_kernel("pck00011.tpc")[0] == str(p)

    def test_a_real_upstream_update_keeps_its_supersession(self, db, tmp_path):
        """The condition is narrow on purpose: clear only when the superseding
        content is findable nowhere. A genuine newer version on disk must keep
        winning, or resolve() could start handing back a real older release."""
        old = tmp_path / "naif0012.tls"
        old.write_text("OLD LSK")
        old_hash = sha256_file(old)
        db.register_file(old, mission="JUICE")

        old.write_text("NEW LSK")                 # upstream updated in place
        db.register_file(old, mission="JUICE")
        new_hash = sha256_file(old)
        assert new_hash != old_hash

        # the older content reappears elsewhere and gets registered
        elsewhere = tmp_path / "copy" / "naif0012.tls"
        elsewhere.parent.mkdir()
        elsewhere.write_text("OLD LSK")
        db.register_file(elsewhere, mission="JUICE")

        row = db.con.execute(
            "SELECT superseded_by FROM kernels WHERE sha256 = ?", [old_hash]
        ).fetchone()
        assert row[0] == new_hash, (
            "supersession cleared even though the newer content is on disk"
        )
        assert db.resolve_kernel("naif0012.tls")[0] == str(old)
