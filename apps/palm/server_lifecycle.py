from threading import Thread
import receiver


def on_server_loaded(_server_context):
    try:
        t = Thread(target=receiver.stream_receive, daemon=True)
        t.start()

    except:
        print("Can't connect to the stream")
