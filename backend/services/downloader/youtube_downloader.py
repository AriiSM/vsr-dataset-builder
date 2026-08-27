"""
YouTube Downloader Service

Downloads YouTube videos at MAXIMUM available quality with Creative Commons
filtering, and guarantees the file that lands in data/raw/ is complete and
readable before the rest of the pipeline ever touches it.

Design contract (IN → OUT):
    IN  : YouTube URL (or bare video id) + CC-license policy
    OUT : {output_dir}/{output_name}.mp4 — merged, remuxed (never re-encoded),
          integrity-verified (video+audio streams present, duration matches
          what YouTube reported)

Structure:
    VideoInfo         — metadata extracted without downloading
    VideoProperties   — real properties of the file on disk (ffprobe)
    YouTubeDownloader — public API: get_video_info / check_creative_commons /
                        download; every step delegated to a single-purpose
                        private helper.
"""

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yt_dlp
from loguru import logger


# ============================================================== exceptions


class EnvironmentNotReadyError(RuntimeError):
    """The machine is missing a required tool (e.g. ffprobe). Retrying can
    never help — surface immediately with setup instructions."""


# ============================================================== data types


@dataclass
class VideoInfo:
    """Metadata extracted from YouTube WITHOUT downloading the video.

    Only fields the pipeline actually consumes: identity, catalog info
    (title/channel/duration), the license gate inputs (license/description),
    and the best available quality (resolution/fps).
    """

    video_id: str
    title: str
    channel: str
    duration: float
    license: str
    description: str
    resolution: str          # best AVAILABLE, e.g. "1920x1080"
    fps: float

    _CC_INDICATORS = (
        "creative commons",
        "cc-by",
        "cc by",
        "creativecommons.org",
    )

    @property
    def is_creative_commons(self) -> bool:
        """True when the license field or description signals a CC license."""
        license_text = (self.license or "").lower()
        description = (self.description or "").lower()
        return any(
            marker in license_text or marker in description
            for marker in self._CC_INDICATORS
        )


@dataclass
class VideoProperties:
    """Real properties of a DOWNLOADED file, read back with ffprobe."""

    width: int
    height: int
    fps: float
    video_codec: str
    duration: float

    def __str__(self) -> str:
        return (
            f"{self.width}x{self.height} {self.video_codec} "
            f"{self.fps:.4g}fps, {self.duration:.1f}s"
        )


# ============================================================== downloader


class YouTubeDownloader:
    """Download YouTube videos with yt-dlp — maximum quality, verified.

    Example:
        downloader = YouTubeDownloader(output_dir="./data/raw")
        info = downloader.get_video_info("https://youtube.com/watch?v=...")
        if info.is_creative_commons:
            path = downloader.download("https://youtube.com/watch?v=...", "md_001")
    """

    # Maximum-quality chain: prefer H.264 at its best available resolution
    # (usually 1080p — fast decode, native mp4); otherwise take the absolute
    # best stream (VP9/AV1 at 1440p/4K), remuxed losslessly into mp4.
    DEFAULT_FORMAT = (
        "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/"
        "bestvideo+bestaudio/best"
    )

    # Downloaded duration may differ from the reported one by at most this
    # fraction before the file is treated as truncated.
    DURATION_TOLERANCE = 0.05

    # ------------------------------------------------------------ lifecycle

    def __init__(
        self,
        output_dir: Path,
        format_string: str = DEFAULT_FORMAT,
        rate_limit: str = "5M",
        max_retries: int = 3,
        retry_delay: int = 5,
        fragment_retries: int = 10,
        sleep_interval: int = 2,
        cookies_file: Optional[Path] = None,
        cookies_from_browser: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._base_options = self._build_base_options(
            format_string=format_string,
            rate_limit_bytes=self._parse_rate_limit(rate_limit),
            max_retries=max_retries,
            fragment_retries=fragment_retries,
            sleep_interval=sleep_interval,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
        )

    # ----------------------------------------------------------- public API

    def get_video_info(self, url: str) -> VideoInfo:
        """Extract metadata (title, license, best resolution) WITHOUT downloading."""
        youtube_id = self._extract_youtube_id(url)
        watch_url = self._watch_url(youtube_id)

        options = {**self._base_options, 'skip_download': True}
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                raw = ydl.extract_info(watch_url, download=False)
        except Exception as e:
            logger.error(f"Failed to get video info for {url}: {e}")
            raise

        resolution, fps = self._best_available_resolution(raw)
        return VideoInfo(
            video_id=youtube_id,
            title=raw.get('title', ''),
            channel=raw.get('channel', raw.get('uploader', '')),
            duration=raw.get('duration', 0),
            license=raw.get('license', ''),
            description=raw.get('description', ''),
            resolution=resolution,
            fps=fps,
        )

    def check_creative_commons(self, url: str) -> tuple[bool, VideoInfo]:
        """Return (is_cc, info) for the license gate."""
        info = self.get_video_info(url)
        return info.is_creative_commons, info

    def download(
        self,
        url: str,
        output_name: str,
        verify_cc: bool = True,
    ) -> Optional[Path]:
        """Download one video to {output_dir}/{output_name}.mp4.

        Orchestration only — each step lives in its own helper:
        license gate → attempt loop (exponential backoff) → locate output →
        lossless remux if needed → integrity verification.

        An existing verified file is never re-downloaded (delete it to force).
        Returns the verified path, or None on failure.
        """
        youtube_id = self._extract_youtube_id(url)
        output_path = self.output_dir / f"{output_name}.mp4"

        if output_path.exists():
            logger.info(f"Video already exists: {output_path}")
            return output_path

        expected_duration = None
        if verify_cc:
            is_cc, info = self.check_creative_commons(url)
            if not is_cc:
                logger.warning(
                    f"Video {youtube_id} is not Creative Commons licensed "
                    f"(license: {info.license!r})"
                )
                return None
            logger.info(f"Verified CC license for: {info.title}")
            expected_duration = info.duration or None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Downloading {youtube_id} (attempt {attempt}/{self.max_retries})"
                )
                properties = self._download_once(
                    youtube_id, output_name, output_path, expected_duration
                )
                logger.info(f"Downloaded successfully: {output_path} [{properties}]")
                return output_path
            except EnvironmentNotReadyError:
                raise  # setup problem — retrying cannot help
            except Exception as e:
                logger.error(f"Download attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # 5s, 10s, 20s…
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

        logger.error(f"Failed to download {youtube_id} after {self.max_retries} attempts")
        return None

    # ------------------------------------------------- download step helpers

    def _download_once(
        self,
        youtube_id: str,
        output_name: str,
        output_path: Path,
        expected_duration: Optional[float],
    ) -> VideoProperties:
        """One complete download attempt. Raises on ANY failure so the
        retry loop stays trivial."""
        options = {
            **self._base_options,
            'outtmpl': str(self.output_dir / f"{output_name}.%(ext)s"),
            'merge_output_format': 'mp4',
            # REMUXER, not convertor: container change only, never re-encoding
            'postprocessors': [{
                'key': 'FFmpegVideoRemuxer',
                'preferedformat': 'mp4',
            }],
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([self._watch_url(youtube_id)])

        produced = self._locate_output(output_name)
        if produced is None:
            raise RuntimeError("download finished but no output file found")

        if produced.suffix != '.mp4':
            if not self._remux_to_mp4(produced, output_path):
                raise RuntimeError(f"lossless remux to mp4 failed for {produced.name}")
            produced.unlink(missing_ok=True)

        properties = self._probe_video(output_path, expected_duration)
        if properties is None:
            # Delete the corrupt file so the next attempt starts clean —
            # a truncated download must fail HERE, loudly, not later in VAD.
            output_path.unlink(missing_ok=True)
            raise RuntimeError("integrity check failed (truncated/corrupt file)")
        return properties

    def _locate_output(self, output_name: str) -> Optional[Path]:
        """Find the file yt-dlp produced, whatever container it chose."""
        for extension in ('mp4', 'mkv', 'webm'):
            candidate = self.output_dir / f"{output_name}.{extension}"
            if candidate.exists():
                return candidate
        return None

    def _remux_to_mp4(self, source: Path, destination: Path) -> bool:
        """Change container to mp4 WITHOUT re-encoding (stream copy)."""
        command = [
            self._find_ffmpeg_tool("ffmpeg"), "-y",
            "-i", str(source),
            "-c", "copy",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True)
        return (
            result.returncode == 0
            and destination.exists()
            and destination.stat().st_size > 0
        )

    def _probe_video(
        self,
        video_path: Path,
        expected_duration: Optional[float],
    ) -> Optional[VideoProperties]:
        """Validate the downloaded file and read its real properties.

        Checks: readable container, BOTH video and audio streams present,
        duration > 0, and (when known) duration within DURATION_TOLERANCE
        of what YouTube reported — a truncated download fails this.
        """
        command = [
            self._find_ffmpeg_tool("ffprobe"), "-v", "quiet",
            "-print_format", "json", "-show_streams", "-show_format",
            str(video_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, timeout=60)
            data = json.loads(result.stdout)
        except FileNotFoundError:
            # Environment problem, not a corrupt file — retrying would fail
            # every video with a misleading "integrity check failed".
            raise EnvironmentNotReadyError(
                "ffprobe is not installed. imageio-ffmpeg bundles only ffmpeg, "
                "NOT ffprobe — install ffmpeg system-wide (winget install "
                "ffmpeg / brew install ffmpeg) so ffprobe is on PATH."
            ) from None
        except Exception as e:
            logger.warning(f"ffprobe failed on {video_path}: {e}")
            return None

        streams = data.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        if video is None or audio is None:
            logger.warning(f"{video_path.name}: missing video or audio stream")
            return None

        duration = float(data.get("format", {}).get("duration", 0) or 0)
        if duration <= 0:
            logger.warning(f"{video_path.name}: zero duration")
            return None
        if expected_duration and (
            abs(duration - expected_duration)
            > expected_duration * self.DURATION_TOLERANCE
        ):
            logger.warning(
                f"{video_path.name}: duration {duration:.1f}s differs from "
                f"expected {expected_duration:.1f}s — probably truncated"
            )
            return None

        return VideoProperties(
            width=video.get("width", 0),
            height=video.get("height", 0),
            fps=self._parse_frame_rate(video.get("avg_frame_rate", "25/1")),
            video_codec=video.get("codec_name", "?"),
            duration=duration,
        )

    # -------------------------------------------------- environment helpers

    def _build_base_options(
        self,
        format_string: str,
        rate_limit_bytes: int,
        max_retries: int,
        fragment_retries: int,
        sleep_interval: int,
        cookies_file: Optional[Path],
        cookies_from_browser: Optional[str],
    ) -> dict:
        """Assemble the yt-dlp options shared by info extraction and download."""
        options = {
            'format': format_string,
            'ratelimit': rate_limit_bytes,
            'retries': max_retries,
            # Resume partial .part files instead of restarting, and survive
            # individual DASH fragments failing mid-download.
            'continuedl': True,
            'fragment_retries': fragment_retries,
            # Politeness pause between consecutive requests/downloads —
            # keeps batch runs under YouTube's bot-detection radar.
            'sleep_interval': sleep_interval,
            'socket_timeout': 30,
            'ignoreerrors': False,
            'no_warnings': False,
            'extract_flat': False,
            'quiet': False,
            'no_color': True,
            # JS runtime for YouTube's "n challenge" (deno/node/bun, whichever
            # is installed) + permission to fetch the challenge-solver script.
            'js_runtimes': self._detect_js_runtime(),
            'remote_components': {'ejs:github'},
        }

        ffmpeg_dir = self._prepare_ffmpeg_for_ytdlp()
        if ffmpeg_dir:
            options['ffmpeg_location'] = ffmpeg_dir
        if cookies_file and Path(cookies_file).exists():
            options['cookiefile'] = str(cookies_file)
        if cookies_from_browser:
            options['cookiesfrombrowser'] = (cookies_from_browser,)
        return options

    @staticmethod
    def _detect_js_runtime() -> dict:
        """Pick the first JS runtime available on PATH (Windows usually has node)."""
        for runtime in ("deno", "node", "bun"):
            if shutil.which(runtime):
                return {runtime: {}}
        return {"deno": {}}  # yt-dlp's default; errors loudly if missing

    @staticmethod
    def _prepare_ffmpeg_for_ytdlp() -> Optional[str]:
        """Expose the bundled imageio ffmpeg under the standard name yt-dlp
        expects ('ffmpeg.exe'), in a temp dir. Returns that dir, or None."""
        try:
            import imageio_ffmpeg
            source = Path(imageio_ffmpeg.get_ffmpeg_exe())
            target_dir = Path(tempfile.gettempdir()) / "_ytdlp_ffmpeg"
            target_dir.mkdir(exist_ok=True)
            target = target_dir / "ffmpeg.exe"
            if not target.exists():
                shutil.copy2(source, target)
            return str(target_dir)
        except Exception:
            return None

    @staticmethod
    def _find_ffmpeg_tool(name: str) -> str:
        """Locate ffmpeg/ffprobe: PATH first, then the imageio bundle."""
        on_path = shutil.which(name)
        if on_path:
            return on_path
        try:
            import imageio_ffmpeg
            bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
            if name == "ffmpeg":
                return str(bundled)
            sibling = bundled.parent / bundled.name.replace("ffmpeg", name)
            if sibling.exists():
                return str(sibling)
        except ImportError:
            pass
        return name  # last resort — subprocess will raise a clear error

    # ----------------------------------------------------- parsing helpers

    @staticmethod
    def _watch_url(youtube_id: str) -> str:
        return f"https://www.youtube.com/watch?v={youtube_id}"

    @staticmethod
    def _extract_youtube_id(url: str) -> str:
        """Accept watch/short/embed URLs or a bare 11-character id."""
        patterns = (
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)'
            r'([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$',
        )
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(f"Could not extract YouTube ID from: {url}")

    @staticmethod
    def _parse_rate_limit(rate: str) -> int:
        """'5M' → bytes/second."""
        multipliers = {'K': 1024, 'M': 1024 ** 2, 'G': 1024 ** 3}
        match = re.match(r'^(\d+)([KMG])?$', rate.upper())
        if match:
            return int(match.group(1)) * multipliers.get(match.group(2), 1)
        return 5 * 1024 ** 2  # default 5 MB/s

    @staticmethod
    def _parse_frame_rate(raw: str) -> float:
        """ffprobe's '25/1' → 25.0 (safe on malformed input)."""
        try:
            numerator, denominator = raw.split("/")
            return float(numerator) / float(denominator) if float(denominator) else 25.0
        except (ValueError, ZeroDivisionError):
            return 25.0

    @staticmethod
    def _best_available_resolution(raw_info: dict) -> tuple[str, float]:
        """Best resolution/fps YouTube offers (formats are listed worst→best,
        so 'first format' would under-report — take the max height)."""
        video_formats = [
            fmt for fmt in raw_info.get('formats', [])
            if fmt.get('vcodec') != 'none' and fmt.get('height')
        ]
        if not video_formats:
            return "unknown", 25.0
        best = max(video_formats, key=lambda fmt: fmt.get('height', 0))
        resolution = f"{best.get('width', '?')}x{best.get('height', '?')}"
        return resolution, best.get('fps', 25.0) or 25.0


# ================================================================= CLI


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download YouTube videos (max quality, verified)")
    parser.add_argument("url", help="YouTube URL or video ID")
    parser.add_argument("-o", "--output", required=True, help="Output filename (without extension)")
    parser.add_argument("-d", "--dir", default="./data/raw", help="Output directory")
    parser.add_argument("--no-cc-check", action="store_true", help="Skip CC verification")
    parser.add_argument("--info-only", action="store_true", help="Only show video info")

    args = parser.parse_args()
    downloader = YouTubeDownloader(output_dir=Path(args.dir))

    if args.info_only:
        video_info = downloader.get_video_info(args.url)
        print(f"\nVideo Info:")
        print(f"  ID:         {video_info.video_id}")
        print(f"  Title:      {video_info.title}")
        print(f"  Channel:    {video_info.channel}")
        print(f"  Duration:   {video_info.duration}s")
        print(f"  Best res:   {video_info.resolution} @ {video_info.fps}fps")
        print(f"  License:    {video_info.license}")
        print(f"  Is CC:      {video_info.is_creative_commons}")
    else:
        result = downloader.download(args.url, args.output, verify_cc=not args.no_cc_check)
        print(f"\nDownloaded: {result}" if result else "\nDownload failed")
