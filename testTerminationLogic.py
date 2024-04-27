from overdrive import Overdrive, OverdriveDelegate
import struct
import threading
import queue
import logging
import bluepy.btle as btle


location = 0
peice = 0
offset = 0
speed = 0
clockwise = 0
lastPeice = 0


class OverdriveAnki(Overdrive):
    
    def __init__(self, addr):
        
        Overdrive._delegate = OverdriveDelegateAnki(self)
        Overdrive.__init__(self, addr)
        self.error = ""

    def _executor(self):
        """Notification thread, for internal use only."""
        data = None
        while Overdrive._connected:
            if self._reconnect:
                while True:
                    try:
                        self.connect()
                        self._reconnect = False
                        if data is not None:
                            self._writeChar.write(data)
                        break
                    except btle.BTLEException as e:
                        logging.getLogger("anki.overdrive").error(e.message)
                        self._reconnect = True
            try:
                data = self._writeQueue.get_nowait()
                self._writeChar.write(data)
                data = None
            except queue.Empty:
                try:
    
                    self._peripheral.waitForNotifications(0.001)
                except btle.BTLEException as e:
                    self.error = e.message
                    print("Error", self.error)
                    logging.getLogger("anki.overdrive").error(e.message)
                    self._reconnect = True
            except btle.BTLEException as e:
                self.error = e.message
                print("Error", self.error)
                logging.getLogger("anki.overdrive").error(e.message)
                self._reconnect = True
        self._disconnect()
        self._btleSubThread = None


class OverdriveDelegateAnki(OverdriveDelegate):
    """Notification delegate object for Bluepy, for internal use only."""

    def __init__(self, OverdriveAnki):

        self.overdrive = OverdriveAnki
        OverdriveDelegate.__init__(self)


    
    def handleNotification(self, handle, data):
        print("The one that we want to call")
        if self.handle == handle:
            self.notificationsRecvd += 1
            (commandId,) = struct.unpack_from("B", data, 1)
            print("CommandID:", commandId)
            if commandId == 0x27:
                # Location position
                location, piece, offset, speed, clockwiseVal = struct.unpack_from("<BBfHB", data, 2)
                clockwise = False
                if clockwiseVal == 0x47:
                    clockwise = True
                threading.Thread(target=self.overdrive._locationChangeCallback, args=(location, piece, speed, clockwise)).start()
            if commandId == 0x29:
                # Transition notification
                piece, piecePrev, offset, direction = struct.unpack_from("<BBfB", data, 2)
                threading.Thread(target=self.overdrive._transitionCallback).start()
            elif commandId == 0x17:
                # Pong
                threading.Thread(target=self.overdrive._pongCallback).start()

def locationChangeCallback(addr, data):
    global location, peice, offset, speed, clockwise
    location = data["location"]
    peice = data["piece"]
    offset = data["offset"]
    speed = data["speed"]


cars = {
    'skull': 'FD:24:03:51:B8:44',
    'shock': 'D1:8A:9D:BB:B6:4F',
    'police': 'FF:B1:ED:98:FF:FF',
    'bigBang': 'F2:2D:0F:19:FD:27'
}

car = OverdriveAnki(cars['bigBang'])
car.setLocationChangeCallback(locationChangeCallback)
car.changeSpeed(800, 250)