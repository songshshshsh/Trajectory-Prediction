import numpy as np
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser(description='preprocess crowd tracking data')
parser.add_argument('--track_path', default='', type=str, metavar='PATH',
	help='path to tracking data (default: none)')

def preprocessIntention():
	args = parser.parse_args()

	tmp = sio.loadmat(args.track_path)
	raw = tmp['trks'].squeeze()

	trks = []
	for i in xrange(raw.shape[0]):
		trk = raw[i]
		intention = np.concatenate((trk[0][trk[2].shape[0] - 1],
							trk[1][trk[2].shape[0] - 1]))
		intention_array = np.tile(intention, (trk[2].shape[0], 1))
		people_id_array = np.empty(shape=(trk[2].shape[0], 1))
		people_id_array.fill(i)
		trks.append(np.concatenate((trk[0], trk[1], trk[2], intention_array[:, 0].reshape((trk[2].shape[0], 1)),
			intention_array[:, 1].reshape((trk[2].shape[0], 1)), people_id_array), axis=1))

	print('Valid trajs {}'.format(len(trks)))
	np.save('trks_intention.npy', trks)

def preprocessTime():
	trks = np.load('trks_intention.npy')
	timeDict = dict()
	for i in xrange(trks.shape[0]):
		if i % 100 == 0:
			print i
		trk = trks[i]
		for j in xrange(trk[0].shape[0]):
			if (int(trk[j][2]) in timeDict.itervalues()):
				timeDict[int(trk[j][2])].append(np.array([trk[j][0], trk[j][1], trk[j][2],
					trk[j][3], trk[j][4], trk[j][5]]))
			else:
				timeDict[int(trk[j][2])] = [np.array([trk[j][0], trk[j][1], trk[j][2],
					trk[j][3], trk[j][4], trk[j][5]])]
	print len(timeDict.itervalues())
	np.save('trks_time.npy', timeDict)


if __name__ == '__main__':
	preprocessIntention()
	preprocessTime()