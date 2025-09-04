from moviepy import ImageSequenceClip, TextClip, CompositeVideoClip
import os
import imageio


def mask_png(frames):
    for frame in frames:
        im = imageio.imread(frame)
        im[im[:, :, 3] < 1, :] = 255
        imageio.imwrite(frame, im[:, :, 0:3])
    return


class Video:
    def __init__(self, frame_path: str, fps: float = 12.5, res="high"):
        frame_path = str(frame_path)
        print(f"--- 步骤 3 (video.py/__init__): Video 对象被创建, 收到的 fps 参数为: {fps}")
        self.fps = fps
        print(f"--- 步骤 4 (video.py/__init__): self.fps 属性被设置为: {self.fps}")

        if res == "low":
            bitrate = "500k"
        else:
            bitrate = "5000k"

        # ✅ 2.x 新版配置
        self._conf = {
            "codec": "libx264",
            "audio_codec": "aac",
            "bitrate": bitrate
        }

        frames = [os.path.join(frame_path, x)
                  for x in sorted(os.listdir(frame_path))]

        # mask background white for videos
        mask_png(frames)

        video = ImageSequenceClip(frames, fps=fps)
        self.video = video
        self.duration = video.duration

    def add_text(self, text):
        # needs ImageMagick
        video_text = TextClip(text,
                              font='Amiri',
                              color='white',
                              method='caption',
                              align="center",
                              size=(self.video.w, None),
                              fontsize=30)
        video_text = video_text.on_color(size=(self.video.w, video_text.h + 5),
                                         color=(0, 0, 0),
                                         col_opacity=0.6)
        video_text = video_text.set_pos('top')

        self.video = CompositeVideoClip([self.video, video_text])

    def save(self, out_path):
        out_path = str(out_path)
        print(f"--- 步骤 5 (video.py/save): 即将调用 write_videofile, self._conf 字典为: {self._conf}")
        self.video.write_videofile(out_path, fps=self.fps, **self._conf)
