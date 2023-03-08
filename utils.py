"""
The code below provides waitkey capability for pyplot is obtained from:
https://gist.github.com/smidm/745b4006a54cfe7343f4a50855547cc3
Thanks the author Matěj Šmíd https://gist.github.com/smidm
"""
import matplotlib.pyplot as plt
closed = False
def handle_close(evt):
    global closed
    closed = True

def waitforbuttonpress():
    while plt.waitforbuttonpress(0.2) is None:
        if closed:
            return False
    return True