from collections import deque
from bsread import source

BUFFER_SIZE = 100
data_buffer = deque(maxlen=BUFFER_SIZE)

state = 'polling'

unstreaked = 'SAROP11-PALMK118:CH1_BUFFER'
streaked = 'SAROP11-PALMK118:CH2_BUFFER'
undulator = 'SARUN15-UIND030:FELPHOTENEBD'
monochrom = 'SAROP11-ODCM105:ENERGY'


def stream_receive():
    global state
    with source(channels=[unstreaked, streaked, undulator, monochrom]) as stream:
        while True:
            message = stream.receive()
            data_buffer.append(message.data.data)
            state = 'receiving'
