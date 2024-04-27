import bluepy.btle as btle
import struct
import time


peripheral = btle.Peripheral()
readChar = None
writeChar = None

cars = {
    'skull': 'FD:24:03:51:B8:44',
    'shock': 'D1:8A:9D:BB:B6:4F',
    'police': 'FF:B1:ED:98:FF:FF',
    'bigBang': 'F2:2D:0F:19:FD:27'
}



addr = cars['bigBang']

peripheral.connect(addr, btle.ADDR_TYPE_RANDOM)

readChar = peripheral.getCharacteristics(1, 0xFFFF, "be15bee06186407e83810bd89c4d8df4")[0]
writeChar = peripheral.getCharacteristics(1, 0xFFFF, "be15bee16186407e83810bd89c4d8df4")[0]

try:
    
    data = b"\x90\x01\x01"
    writeChar.write(data)

    time.time.sleep(2)

    command = struct.pack("<BHHB", 0x24, 300, 250, 0x01)
    writeChar.write(command)

except:
    pass