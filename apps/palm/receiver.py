from collections import deque
from bsread import source

BUFFER_SIZE = 100
data_buffer = deque(maxlen=BUFFER_SIZE)

state = 'polling'

# bernina
# reference = 'SAROP21-PALMK134:CH1_BUFFER'
# streaked = 'SAROP21-PALMK134:CH2_BUFFER'

# alvra
reference = 'SAROP11-PALMK118:CH1_BUFFER'
streaked = 'SAROP11-PALMK118:CH2_BUFFER'

def stream_receive():
    global state
    with source(channels=[reference, streaked]) as stream:
        while True:
            message = stream.receive()
            data_buffer.append(message.data.data)
            state = 'receiving'
