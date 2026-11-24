import argparse
import os
import shutil
import subprocess
import sys


def jobs(source, target):
	for folder, _, files in os.walk(source):
		out = os.path.join(target, os.path.relpath(folder, source))
		for name in sorted(files):
			stem, ext = os.path.splitext(name)
			ext = ext.lower()
			if ext == '.wav':
				yield os.path.join(folder, name), os.path.join(out, stem + '.ogg')
			elif ext == '.json':
				yield os.path.join(folder, name), os.path.join(out, name)


def fresh(src, dst):
	return os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src)


def encode(src, dst, quality):
	command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', src, '-map_metadata', '-1', '-c:a', 'libvorbis', '-q:a', str(quality), dst]
	return subprocess.run(command).returncode == 0


def main():
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	parser = argparse.ArgumentParser(prog='convert_music')
	parser.add_argument('--source', default=os.path.join(root, 'assets', 'music'))
	parser.add_argument('--target', default=os.path.join(root, 'assets', 'music_ogg'))
	parser.add_argument('--quality', type=float, default=5)
	parser.add_argument('--force', action='store_true')
	options = parser.parse_args()
	if not os.path.isdir(options.source):
		sys.exit('no source folder: %s' % options.source)
	failed = 0
	for src, dst in jobs(options.source, options.target):
		if not options.force and fresh(src, dst):
			continue
		os.makedirs(os.path.dirname(dst), exist_ok=True)
		if dst.endswith('.json'):
			shutil.copy2(src, dst)
			continue
		print(os.path.relpath(dst, options.target), flush=True)
		if not encode(src, dst, options.quality):
			failed += 1
			print('  failed', flush=True)
	sys.exit(1 if failed else 0)


if __name__ == '__main__':
	main()
