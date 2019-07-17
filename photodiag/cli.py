import argparse
import logging
import os

from bokeh.application.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.server.server import Server

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """The photodiag command line interface.

    This is a wrapper around bokeh server that provides an interface to launch
    applications bundled with the photodiag package.
    """
    parser = argparse.ArgumentParser(
        prog='photodiag', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'app',
        type=str,
        choices=['palm'],
        help="photodiag application",
    )

    parser.add_argument(
        '--port', type=int, default=5006, help="the port to listen on for HTTP requests"
    )

    parser.add_argument(
        '--allow-websocket-origin',
        metavar='HOST[:PORT]',
        type=str,
        action='append',
        default=None,
        help="hostname that can connect to the server websocket",
    )

    parser.add_argument(
        '--args',
        nargs=argparse.REMAINDER,
        default=[],
        help="command line arguments for the photodiag application",
    )

    args = parser.parse_args()

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'apps', args.app)
    logger.info(app_path)

    handler = DirectoryHandler(filename=app_path, argv=args.args)
    server = Server(
        {'/': Application(handler)},
        port=args.port,
        allow_websocket_origin=args.allow_websocket_origin,
    )

    server.start()
    server.io_loop.start()


if __name__ == "__main__":
    main()
