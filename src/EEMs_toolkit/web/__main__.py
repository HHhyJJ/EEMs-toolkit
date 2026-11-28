"""``eems-web`` entry point: start uvicorn and open the browser."""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from pathlib import Path


def _static_ready() -> bool:
    return (Path(__file__).parent / 'static' / 'index.html').is_file()


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault('MPLBACKEND', 'Agg')
    parser = argparse.ArgumentParser(prog='eems-web', description='EEMs-toolkit local web interface')
    parser.add_argument('--host', default='127.0.0.1', help='Bind address (default 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, help='Port (default 8000)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open a browser window')
    parser.add_argument('--reload', action='store_true', help='Reload on code changes (development only)')
    args = parser.parse_args(argv)

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        raise SystemExit(
            'The web extras are not installed. Run: pip install "EEMs-toolkit[web]"'
        )
    if not _static_ready():
        print(
            'Warning: frontend assets are missing (EEMs_toolkit/web/static/index.html).\n'
            'Build them with: cd frontend && npm install && npm run build'
        )

    url = f'http://{args.host if args.host != "0.0.0.0" else "127.0.0.1"}:{args.port}/'
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f'EEMs-toolkit web interface: {url}')
    import uvicorn
    uvicorn.run(
        'EEMs_toolkit.web.app:app',
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level='info',
    )


if __name__ == '__main__':
    main()
