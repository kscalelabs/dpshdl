"""Functions for managing experiments."""

import contextlib
import hashlib
import itertools
import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import requests

USER_AGENT = "dpshdl"

logger = logging.getLogger(__name__)


def check_md5(file_path: str | Path, hash_str: str | None, chunk_size: int = 2**16) -> bool:
    """Checks the MD5 of the downloaded file.

    Args:
        file_path: Path to the downloaded file.
        hash_str: Expected MD5 of the file; if None, return True.
        chunk_size: Size of the chunks to read from the file.

    Returns:
        True if the MD5 matches, False otherwise.
    """
    if hash_str is None:
        return True

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)

    return md5.hexdigest() == hash_str


def check_sha256(file_path: str | Path, hash_str: str | None, chunk_size: int = 2**16) -> bool:
    """Checks the SHA256 of the downloaded file.

    Args:
        file_path: Path to the downloaded file.
        hash_str: Expected SHA256 of the file; if None, return True.
        chunk_size: Size of the chunks to read from the file.

    Returns:
        True if the SHA256 matches, False otherwise.
    """
    if hash_str is None:
        return True

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)

    return sha256.hexdigest() == hash_str


class FileDownloader:
    """Provides a simple interface for downloading URLs.

    This class is meant to be subclassed to provide different download
    locations. For example, when downloading pretrained models, use the
    :class:`ModelDownloader` class.

    Typically, you should simply use the :func:`ensure_downloaded` function
    to make sure the file is downloaded to the correct location.

    This is adapted in large part from the reference implementation in the
    Torchvision library.

    Parameters:
        url: The URL to download from.
        dnames: The directory names to download to.
        md5: The expected MD5 of the file.
        sha256: The expected SHA256 of the file.
        is_tmp: Whether to download to a temporary directory.
        recheck_hash: Whether to recheck the hash after downloading.
        max_redirect_hops: The maximum number of redirects to follow.
    """

    def __init__(
        self,
        url: str,
        *dnames: str,
        md5: str | None = None,
        sha256: str | None = None,
        root_dir: Path | None = None,
        is_tmp: bool = False,
        recheck_hash: bool = False,
        max_redirect_hops: int = 3,
    ) -> None:
        super().__init__()

        assert len(dnames) >= 1, "Must provide at least 1 directory name"
        if is_tmp:
            filepath = Path(tempfile.mkdtemp("models"))
        else:
            filepath = Path(root_dir or Path.cwd() / "data")
        for dname in dnames:
            filepath = filepath / dname
        (root := filepath.parent).mkdir(parents=True, exist_ok=True)

        self.url = url
        self.filename = filepath.name
        self.root = root
        self.md5 = md5
        self.sha256 = sha256
        self.recheck_hash = recheck_hash
        self.max_redirect_hops = max_redirect_hops

    @property
    def filepath(self) -> Path:
        return self.root / self.filename

    @property
    def is_downloaded(self) -> bool:
        if not self.filepath.exists():
            return False
        if self.recheck_hash and not self.check_hashes():
            logger.warning("A file was found for %s in %s, but its hashes do not match.", self.url, self.filepath)
            self.filepath.unlink()
            return False
        return True

    def check_hashes(self) -> bool:
        return check_sha256(self.filepath, self.sha256) and check_md5(self.filepath, self.md5)

    def ensure_downloaded(self) -> Path:
        """Ensures the file is downloaded and returns the path to it.

        By default, we only check the hash once when the file is downloaded,
        and we don't bother rechecking unless ``recheck_hash`` is set to True.

        Returns:
            The path to the downloaded file.
        """
        if not self.is_downloaded:
            self.download()
            if not self.check_hashes():
                self.filepath.unlink()
                raise RuntimeError(f"Hashes for {self.filepath} do not match. The corruped file has been deleted.")
        return self.filepath

    def download(self) -> None:
        root = self.root.expanduser()
        root.mkdir(parents=True, exist_ok=True)

        # Expands the redirect chain if needed.
        url = self._get_redirect_url(self.url, max_hops=self.max_redirect_hops)

        # Checks if file is located on Google Drive.
        file_id = self._get_google_drive_file_id(url)
        if file_id is not None:
            return self.download_file_from_google_drive(file_id, root, self.filename)

        # Downloads the file.
        try:
            logger.info("Downloading %s to %s", url, self.filepath)
            self._urlretrieve(url, self.filepath)
        except (urllib.error.URLError, OSError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                logger.warning("Download failed. Trying HTTP instead of HTTPS: %s to %s", url, self.filepath)
                self._urlretrieve(url, self.filepath)
            else:
                raise e

    @classmethod
    def _save_response_content(cls, content: Iterator[bytes], destination: Path) -> None:
        with open(destination, "wb") as fh:
            for chunk in content:
                if not chunk:  # Filter out keep-alive new chunks.
                    continue
                fh.write(chunk)

    @classmethod
    def _urlretrieve(cls, url: str, filename: Path, chunk_size: int = 1024 * 32) -> None:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            cls._save_response_content(iter(lambda: response.read(chunk_size), b""), filename)

    @classmethod
    def _extract_gdrive_api_response(
        cls,
        response: requests.Response,
        chunk_size: int = 32 * 1024,
    ) -> tuple[str | None, Iterator[bytes]]:
        content = response.iter_content(chunk_size)
        first_chunk = None
        while not first_chunk:  # Filter out keep-alive new chunks.
            first_chunk = next(content)
        content = itertools.chain([first_chunk], content)

        try:
            match = re.search("<title>Google Drive - (?P<api_response>.+?)</title>", first_chunk.decode())
            api_response = match["api_response"] if match is not None else None
        except UnicodeDecodeError:
            api_response = None
        return api_response, content

    @classmethod
    def download_file_from_google_drive(cls, file_id: str, root: Path, filename: str | None = None) -> None:
        root = root.expanduser()
        if not filename:
            filename = file_id
        fpath = root / filename
        root.mkdir(parents=True, exist_ok=True)

        url = "https://drive.google.com/uc"
        params = dict(id=file_id, export="download")
        with requests.Session() as session:
            response = session.get(url, params=params, stream=True)

            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    token = value
                    break
            else:
                api_response, content = cls._extract_gdrive_api_response(response)
                token = "t" if api_response == "Virus scan warning" else None

            if token is not None:
                response = session.get(url, params=dict(params, confirm=token), stream=True)
                api_response, content = cls._extract_gdrive_api_response(response)

            if api_response == "Quota exceeded":
                raise RuntimeError(
                    f"The daily quota of the file {filename} is exceeded and it "
                    f"can't be downloaded. This is a limitation of Google Drive "
                    f"and can only be overcome by trying again later."
                )

            cls._save_response_content(content, fpath)

        # In case we deal with an unhandled GDrive API response, the file should be smaller than 10kB with only text.
        if os.stat(fpath).st_size < 10 * 1024:
            with contextlib.suppress(UnicodeDecodeError), open(fpath) as fh:
                text = fh.read()

                # Regular expression to detect HTML. Copied from https://stackoverflow.com/a/70585604
                if re.search(r"</?\s*[a-z-][^>]*\s*>|(&(?:[\w\d]+|#\d+|#x[a-f\d]+);)", text):
                    warnings.warn(
                        f"We detected some HTML elements in the downloaded file. "
                        f"This most likely means that the download triggered an unhandled API response by GDrive. "
                        f"Please report this to torchvision at https://github.com/pytorch/vision/issues including "
                        f"the response:\n\n{text}"
                    )

    @classmethod
    def _get_google_drive_file_id(cls, url: str) -> str | None:
        parts = urlparse(url)
        if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
            return None
        match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
        if match is None:
            return None
        return match.group("id")

    @classmethod
    def _get_redirect_url(cls, url: str, max_hops: int = 3) -> str:
        initial_url = url
        headers = {"Method": "HEAD", "User-Agent": USER_AGENT}
        for _ in range(max_hops + 1):
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
                if response.url == url or response.url is None:
                    return url
                url = response.url
        raise RecursionError(f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect was {url}.")
