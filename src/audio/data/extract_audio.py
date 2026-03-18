"""
Извлечение аудио из видеофайлов и нарезка на клипы.

Использует ffmpeg для извлечения аудио и нарезки длинных записей.
"""

import os
import sys
import subprocess as sp
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import argparse
from tqdm import tqdm


# Флаг для Windows
IS_WINDOWS = sys.platform == "win32"


def _run_command(cmd, capture_output=True, check=True, quiet=True):
    """Запуск команды с обработкой для Windows."""
    if IS_WINDOWS:
        # На Windows используем os.system для обхода проблем с кодировкой
        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
        else:
            cmd_str = cmd
        
        # Добавляем -quiet для ffmpeg
        if quiet and "ffmpeg" in cmd_str or "ffprobe" in cmd_str:
            # Вставляем -quiet после команды
            parts = cmd_str.split()
            if "-v" not in parts and "-quiet" not in parts:
                parts.insert(1, "-quiet")
                cmd_str = " ".join(parts)
        
        if capture_output:
            result = sp.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                stdin=sp.DEVNULL,
            )
        else:
            ret = os.system(cmd_str)
            result = sp.CompletedProcess(cmd_str, ret, "", "")
        
        if check and result.returncode != 0:
            raise sp.CalledProcessError(result.returncode, cmd_str, result.stdout, result.stderr)
        return result
    else:
        return sp.run(cmd, capture_output=capture_output, check=check, text=True)


@dataclass
class AudioClip:
    """Информация об аудио-клипе."""
    source_video: Path
    output_path: Path
    start_time: float
    duration: float
    clip_index: int


def _find_ffmpeg() -> Optional[str]:
    """Поиск ffmpeg в стандартных путях."""
    # Проверяем PATH
    try:
        result = sp.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return "ffmpeg"
    except (sp.CalledProcessError, FileNotFoundError):
        pass
    
    # Стандартные пути Windows
    search_paths = [
        # WinGet packages
        Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
        # Chocolatey
        Path("C:/ProgramData/chocolatey/bin"),
        # Scoop
        Path.home() / "scoop/apps/ffmpeg/current/bin",
        # Gyan.dev builds
        Path("C:/Program Files/FFmpeg/bin"),
    ]
    
    for base_path in search_paths:
        if base_path.exists():
            for ffmpeg_path in base_path.rglob("ffmpeg.exe"):
                return str(ffmpeg_path)
    
    return None


def check_ffmpeg() -> bool:
    """Проверка доступности ffmpeg."""
    return _find_ffmpeg() is not None


def _find_ffprobe() -> Optional[str]:
    """Поиск ffprobe в стандартных путях."""
    search_paths = [
        Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
        Path("C:/ProgramData/chocolatey/bin"),
        Path.home() / "scoop/apps/ffmpeg/current/bin",
        Path("C:/Program Files/FFmpeg/bin"),
    ]
    
    for base_path in search_paths:
        if base_path.exists():
            for ffprobe_path in base_path.rglob("ffprobe.exe"):
                return str(ffprobe_path)
    
    return None


def _get_ffprobe_cmd() -> str:
    """Получение команды ffprobe с полным путём если нужно."""
    ffprobe_path = _find_ffprobe()
    if ffprobe_path:
        return ffprobe_path
    return "ffprobe"


def get_video_duration(video_path: Path) -> float:
    """
    Получение длительности видеофайла в секундах.
    
    Args:
        video_path: Путь к видеофайлу
        
    Returns:
        Длительность в секундах
    """
    cmd = [
        _get_ffprobe_cmd(),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    if IS_WINDOWS:
        cmd = " ".join(cmd)
    result = _run_command(cmd, capture_output=True, check=True)
    return float(result.stdout.strip())


def _get_ffmpeg_cmd() -> str:
    """Получение команды ffmpeg с полным путём если нужно."""
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path:
        return ffmpeg_path
    return "ffmpeg"


def extract_audio_from_video(
    video_path: Path,
    output_audio_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Извлечение аудио из видеофайла.

    Args:
        video_path: Путь к видеофайлу
        output_audio_path: Путь для сохранения аудио
        sample_rate: Частота дискретизации
        channels: Количество каналов (1 для моно)

    Returns:
        Путь к сохранённому аудиофайлу
    """
    # Создаём директорию если не существует
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        _get_ffmpeg_cmd(),
        "-i", str(video_path),
        "-vn",  # Без видео
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-y",  # Перезаписать если существует
        str(output_audio_path),
    ]
    
    if IS_WINDOWS:
        cmd = " ".join(cmd)
    
    _run_command(cmd, capture_output=False, check=True)
    return output_audio_path


def slice_audio(
    input_audio_path: Path,
    output_dir: Path,
    clip_duration: float,
    overlap: float = 0.0,
    prefix: str = "clip",
    sample_rate: int = 16000
) -> List[AudioClip]:
    """
    Нарезка аудиофайла на клипы фиксированной длины.
    
    Args:
        input_audio_path: Путь к исходному аудио
        output_dir: Директория для сохранения клипов
        clip_duration: Длительность одного клипа в секундах
        overlap: Перекрытие между клипами в секундах
        prefix: Префикс для имён файлов
        
    Returns:
        Список AudioClip с информацией о клипах
    """
    duration = get_video_duration(input_audio_path)
    clips = []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clip_index = 0
    start_time = 0.0
    
    while start_time < duration:
        # Вычисляем фактическую длительность (может быть меньше для последнего клипа)
        actual_duration = min(clip_duration, duration - start_time)
        
        # Формируем имя файла
        output_filename = f"{prefix}_{clip_index:04d}.wav"
        output_path = output_dir / output_filename
        
        # Команда ffmpeg для вырезки сегмента
        cmd = [
            _get_ffmpeg_cmd(),
            "-i", str(input_audio_path),
            "-ss", str(start_time),
            "-t", str(actual_duration),
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-y",
            str(output_path),
        ]
        
        if IS_WINDOWS:
            cmd = " ".join(cmd)

        _run_command(cmd, capture_output=False, check=True)

        clips.append(AudioClip(
            source_video=input_audio_path,
            output_path=output_path,
            start_time=start_time,
            duration=actual_duration,
            clip_index=clip_index,
        ))
        
        clip_index += 1
        start_time += clip_duration - overlap
        
        # Если перекрытие больше или равно длительности, двигаемся на 1 сек
        if overlap >= clip_duration:
            start_time += 1
    
    return clips


def process_video(
    video_path: Path,
    output_root: Path,
    clip_duration: float = 5.0,
    overlap: float = 0.0,
    sample_rate: int = 16000,
    keep_original_audio: bool = False,
) -> List[AudioClip]:
    """
    Обработка видеофайла: извлечение аудио и нарезка на клипы.
    
    Args:
        video_path: Путь к видеофайлу
        output_root: Корневая директория для вывода
        clip_duration: Длительность одного клипа
        overlap: Перекрытие между клипами
        sample_rate: Частота дискретизации
        keep_original_audio: Сохранять ли исходное аудио
        
    Returns:
        Список созданных AudioClip
    """
    # Создаём директорию для этого видео
    video_name = video_path.stem
    video_output_dir = output_root / video_name
    audio_output_dir = video_output_dir / "clips"
    
    # Временный файл для полного аудио
    temp_audio_path = video_output_dir / "full_audio.wav"
    
    try:
        # 1. Извлекаем аудио
        extract_audio_from_video(
            video_path=video_path,
            output_audio_path=temp_audio_path,
            sample_rate=sample_rate,
        )
        
        # 2. Нарезаем на клипы
        clips = slice_audio(
            input_audio_path=temp_audio_path,
            output_dir=audio_output_dir,
            clip_duration=clip_duration,
            overlap=overlap,
            prefix=video_name,
        )
        
        # 3. Удаляем временное аудио если не нужно сохранять
        if not keep_original_audio and temp_audio_path.exists():
            temp_audio_path.unlink()
        
        return clips
        
    except Exception as e:
        # Очистка при ошибке
        if temp_audio_path.exists():
            temp_audio_path.unlink()
        raise e


def process_video_batch(
    video_dir: Path,
    output_root: Path,
    clip_duration: float = 5.0,
    overlap: float = 0.0,
    sample_rate: int = 16000,
    extensions: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov", ".webm"),
    keep_original_audio: bool = False,
    verbose: bool = True,
) -> Tuple[List[AudioClip], List[Tuple[Path, str]]]:
    """
    Пакетная обработка всех видео в директории.
    
    Args:
        video_dir: Директория с видеофайлами
        output_root: Корневая директория для вывода
        clip_duration: Длительность одного клипа
        overlap: Перекрытие между клипами
        sample_rate: Частота дискретизации
        extensions: Поддерживаемые расширения видео
        keep_original_audio: Сохранять ли исходное аудио
        verbose: Выводить ли прогресс
        
    Returns:
        Кортеж из (список AudioClip, список ошибок)
    """
    all_clips = []
    errors = []
    
    # Находим все видеофайлы
    video_files = []
    for ext in extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"[WARN] No video files found in {video_dir}")
        return all_clips, errors
    
    # Обрабатываем каждое видео
    iterator = tqdm(video_files, desc="Processing videos") if verbose else video_files
    
    for video_path in iterator:
        try:
            clips = process_video(
                video_path=video_path,
                output_root=output_root,
                clip_duration=clip_duration,
                overlap=overlap,
                sample_rate=sample_rate,
                keep_original_audio=keep_original_audio,
            )
            all_clips.extend(clips)
            
            if verbose:
                tqdm.write(f"[OK] {video_path.name}: {len(clips)} clips")
                
        except Exception as e:
            error_msg = str(e)
            errors.append((video_path, error_msg))
            if verbose:
                tqdm.write(f"[ERR] {video_path.name}: {error_msg}")
    
    return all_clips, errors


def create_cli_parser() -> argparse.ArgumentParser:
    """Создание CLI парсера аргументов."""
    parser = argparse.ArgumentParser(
        description="Извлечение аудио из видео и нарезка на клипы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Обработать папку с видео
  python extract_audio.py process ./videos ./output
  
  # Нарезать на 10-секундные клипы с перекрытием 2 сек
  python extract_audio.py process ./videos ./output --clip-duration 10 --overlap 2
  
  # Проверить ffmpeg
  python extract_audio.py check
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Команда")
    
    # Команда process
    process_parser = subparsers.add_parser(
        "process",
        help="Обработка видеофайлов",
    )
    process_parser.add_argument(
        "video_dir",
        type=Path,
        help="Директория с видеофайлами",
    )
    process_parser.add_argument(
        "output_dir",
        type=Path,
        help="Директория для сохранения результатов",
    )
    process_parser.add_argument(
        "--clip-duration",
        type=float,
        default=5.0,
        help="Длительность клипа в секундах (по умолчанию: 5.0)",
    )
    process_parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Перекрытие между клипами в секундах (по умолчанию: 0.0)",
    )
    process_parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Частота дискретизации (по умолчанию: 16000)",
    )
    process_parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Сохранять исходное аудио после нарезки",
    )
    process_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Не выводить прогресс",
    )
    
    # Команда check
    subparsers.add_parser(
        "check",
        help="Проверка доступности ffmpeg",
    )
    
    # Команда slice (для готового аудио)
    slice_parser = subparsers.add_parser(
        "slice",
        help="Нарезка аудиофайла на клипы",
    )
    slice_parser.add_argument(
        "input_audio",
        type=Path,
        help="Путь к аудиофайлу",
    )
    slice_parser.add_argument(
        "output_dir",
        type=Path,
        help="Директория для клипов",
    )
    slice_parser.add_argument(
        "--clip-duration",
        type=float,
        default=5.0,
        help="Длительность клипа (по умолчанию: 5.0)",
    )
    slice_parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Перекрытие между клипами (по умолчанию: 0.0)",
    )
    
    return parser


def main():
    """Точка входа CLI."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == "check":
        if check_ffmpeg():
            print("[OK] ffmpeg доступен")
            # Проверяем ffprobe
            try:
                sp.run(
                    [_get_ffprobe_cmd(), "-version"],
                    capture_output=True,
                    check=True,
                )
                print("[OK] ffprobe доступен")
            except (sp.CalledProcessError, FileNotFoundError):
                print("[WARN] ffprobe не найден (требуется для получения длительности)")
        else:
            print("[ERR] ffmpeg не найден!")
            print("Установите ffmpeg:")
            print("  Windows: choco install ffmpeg")
            print("  Linux: sudo apt install ffmpeg")
            print("  macOS: brew install ffmpeg")
    
    elif args.command == "process":
        if not check_ffmpeg():
            print("[ERR] ffmpeg не найден! Установите ffmpeg.")
            return
        
        print(f"Обработка видео из: {args.video_dir}")
        print(f"Результат в: {args.output_dir}")
        print(f"Длительность клипа: {args.clip_duration} сек")
        if args.overlap > 0:
            print(f"Перекрытие: {args.overlap} сек")
        
        clips, errors = process_video_batch(
            video_dir=args.video_dir,
            output_root=args.output_dir,
            clip_duration=args.clip_duration,
            overlap=args.overlap,
            sample_rate=args.sample_rate,
            keep_original_audio=args.keep_original,
            verbose=not args.quiet,
        )
        
        print(f"\n{'='*50}")
        print(f"Всего клипов создано: {len(clips)}")
        if errors:
            print(f"Ошибок: {len(errors)}")
            for path, error in errors:
                print(f"  - {path.name}: {error}")
    
    elif args.command == "slice":
        if not check_ffmpeg():
            print("[ERR] ffmpeg не найден!")
            return
        
        print(f"Нарезка аудио: {args.input_audio}")
        clips = slice_audio(
            input_audio_path=args.input_audio,
            output_dir=args.output_dir,
            clip_duration=args.clip_duration,
            overlap=args.overlap,
        )
        print(f"Создано клипов: {len(clips)}")


if __name__ == "__main__":
    main()
