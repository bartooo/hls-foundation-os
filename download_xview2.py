import wget
import tarfile

def extract_all_files(tar_file_path, extract_to):
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(extract_to)

# train_url = "https://download.xview2.org/train_images_labels_targets.tar.gz?Expires=1696692153&Signature=R~MbKBXmiosPhyfa1TvH3MG1ncvSqV9ymGR-zMzqhvRsUW~qoO4~u~7Xd3wRJf4gaMhbWPiE76Dqf1PlW6G-LaW8aUnVADw75trRWJmsY4WxFQ4jVPHayN1DrjlpIZ5DytxcRvOljVBFbL~krUmNWfTSTRIrdMS4r3020ofnQvaGW0nutOG1aT0yGCgHVsGyKmfMxBEhUSwRMKcYRlm95Lr0RjTvK8SNIYj~0hvnwJ2dTaWVI8QnW8gIH55Q-r9Fn5x8tTCctztVfWfw5rJh04ejGo32z2fMWfpmxlFfEuz6uWnu9DZXM63o6RtJ-yMd~N1xBrmq~FdafOuT3ug2QA__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ"
# val_url = "https://download.xview2.org/test_images_labels_targets.tar.gz?Expires=1696719221&Signature=m~71in2I~1AZiWaRodhOkVwdypk7BHSR561EL~hXthtl1Xnr9XqMtMuAxU3blLhc0L3JWbp9hHVwSW2~r5EEW0xSNMML7K5cxO7aSk9zHjiZQRnd5oHNYpY7LHnOW67inSbkKG5y1oTF5kmLdISBVLd7zKSrwe5OWibFpTjKiwSzlOGsrXY60KnHnY3K9hdIT7VKMvN59m6hv~dvL8cCvmxpJlz4Y7-9HvN0J0j8bCTFyGvxWfk2Unq7D~O2GU33DOaxhpDzqy0sEvnranb-cXpKsTeZdZoNz3JHOhes55xI5qB-UpkJ~pwP9wap~vg4siYkZpW2kSyqKVfbKaGVjg__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ"
test_url = "https://download.xview2.org/tier3.tar.gz?Expires=1696745986&Signature=oobP0eV2aVccHUBMgzet0amMBxuzpCLCNJIni8-x1-NcZnm28GCTTDWViGQ1JbGEohUp2~jmLg3NkRQYITEro9aTt1U1DL~5HVSDLFEx450THb7NDuQapNGVx1ZJojUxm1KuopoSVxqHY1DyPxLrjXYp1fVMztKeJ6NGxKrr2vE53ZOMIy0GYjuj1p34XN7KhpgyQmqukiD9d2POzUoEAj9kNorMjjp7ebWtZv-68yS1i9dnRheR8MgkecS7BriIAPU9k8oN2FVotJ1dxWEXn01gr5OVhEDTounH91bpEAVigMtTOy9SoBX9w7HAf1-upbaFUJXfAUrQJqH52Q1ifg__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ"
wget.download(test_url)
# extract_all_files('train/xview_geotransforms.json.tgz', 'train/')