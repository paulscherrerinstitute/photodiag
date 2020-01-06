import argparse
import logging
from collections import deque

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("beamline")
args = parser.parse_args()

if args.beamline == "alvra":
    reference = "SAROP11-PALMK118:CH1_BUFFER"
    streaked = "SAROP11-PALMK118:CH2_BUFFER"

elif args.beamline == "bernina":
    reference = "SAROP21-PALMK134:CH1_BUFFER"
    streaked = "SAROP21-PALMK134:CH2_BUFFER"

else:
    raise RuntimeError(f"{args.beamline} - unknown beamline")


BUFFER_SIZE = 100
data_buffer = deque(maxlen=BUFFER_SIZE)

state = "polling"


def stream_receive():
    global state
    try:
        from bsread import source
    except ImportError:
        state = "stopped"
        logger.info("bsread is not available")
        return

    try:
        with source(channels=[reference, streaked]) as stream:
            while True:
                message = stream.receive()
                data_buffer.append(message.data.data)
                state = "receiving"

    except Exception:
        logger.exception("can not read from stream")
