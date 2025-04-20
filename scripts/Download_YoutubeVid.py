# script to download videos with their link
#pip install yt-dlp 
import yt_dlp

def download_video(url, output_path='.'):
    # Configure download options
    ydl_opts = {
        'format': 'best',  # Download the best available quality
        'outtmpl': output_path + '/%(title)s.%(ext)s',  # Save with the video title as filename
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Starting download...")
            ydl.download([url])
            print("Download completed!")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == '__main__':
    # Replace with video URL
    video_url = "https://www.youtube.com/watch?v=062nwslzyOQ"
    download_video(video_url)
