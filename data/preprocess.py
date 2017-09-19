import scipy.io as sio
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='preprocess crowd tracking data')
parser.add_argument('--track_path', default='', type=str, metavar='PATH',
                    help='path to tracking data (default: none)')
parser.add_argument('--min_len', default=8, type=int, metavar='N',
                    help='minimum seconds of trajectories to be considered')
parser.add_argument('--fps', default=2., type=float, metavar='N',
                    help='sampling frame rate of trajectories')

fps_video = 23


def preprocess4LSTM():
	args = parser.parse_args()

	tmp = sio.loadmat(args.track_path)
	raw = tmp['trks'].squeeze() # (42821,)

	trks = []
	for i in xrange(raw.shape[0]):
		trk = raw[i]
		if trk[2].shape[0] >= args.min_len * fps_video: # long enough
			if trk[2][-1] - trk[2][0] + 1 != trk[2].shape[0]:
				print('trk {} has {} elements but {} frames'.format(i, trk[2].shape[0], trk[2][-1] - trk[2][0] + 1))

			trks.append(np.concatenate((trk[0], trk[1], trk[2]), axis=1))

	print('Valid trajectories {}'.format(len(trks)))

	trks_sampled = []
	for trk in trks:
		idx = np.arange(0, trk.shape[0], int(fps_video / args.fps))
		trks_sampled.append(trk[idx, :2])

	np.save('trks_min{:0d}_fps{:.1f}.npy'.format(args.min_len, args.fps), trks_sampled)





if __name__ == '__main__':
    preprocess4LSTM()