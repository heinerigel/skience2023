from os.path import normpath
from obspy import read

# example file: ../data/waveforms/38443535/APL.CI.HNN..2019.185.182759.38443535.ms
DATA = normpath('../data/waveforms/{}/{}.{}.{}.{}.*.ms')


def get_data(network, station, location, channel, starttime, endtime, event):
    evid = str(event.resource_id).split('/')[-1]
    fname = DATA.format(evid, station, network, channel, location)
    stream = read(fname, 'MSEED')
    stream.trim(starttime, endtime)
    if not stream:
        print('no data for %s.%s' % (network, station))
    return stream
