import os
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from datasets.video_capture import VideoCapture



class ACTNETDataset(Dataset):
    """
        videos_dir: directory where all videos are stored
        config: AllConfig object
        split_type: 'train' / 'test'
        img_transforms: Composition of transforms
    """

    def __init__(self, config, data, model,
                 split_type='train', img_transforms=None):
        self.config = config
        self.videos_train_dir = os.path.join(config.videos_dir, 'train')
        self.videos_val_dir = os.path.join(config.videos_dir, 'val')
        self.img_transforms = img_transforms
        self.data = data
        self.split_type = split_type
        self.err = 0

        # --------------------------------------------------
        # collect all video ids
        # --------------------------------------------------
        self.videos = [
            video_id for category in self.data.values()
            for video_id in category
        ]

        # --------------------------------------------------
        # video_id -> category_name mapping
        # --------------------------------------------------
        self.video_to_category = {}
        for category, videos in self.data.items():
            for video_id in videos:
                # ACTNET clip-level naming
                self.video_to_category[video_id + '_1'] = category

        # --------------------------------------------------
        # category_name <-> category_id mapping
        # --------------------------------------------------
        all_categories = list(self.data.keys())
        unique_categories = sorted(set(all_categories))

        self.category2id = {
            cat: idx for idx, cat in enumerate(unique_categories)
        }
        self.id2category = {
            idx: cat for cat, idx in self.category2id.items()
        }
        # --------------------------------------------------

        # --------------------------------------------------
        # load captions & construct pairs
        # --------------------------------------------------
        if self.split_type == 'train':
            db_file = '/mnt/data/wang_shaokun/CTVR/datasets/ACTNET/train_queries.json'
            self.vid2caption = load_json(db_file)
            self._construct_all_train_pairs('anet_clip')
        else:
            db_file = '/mnt/data/wang_shaokun/CTVR/datasets/ACTNET/val_queries.json'
            self.vid2caption = load_json(db_file)
            self._construct_all_test_pairs('anet_clip')

    def __getitem__(self, index):
        if self.split_type == 'train':
            video_path, caption, video_id = \
                self._get_vidpath_and_caption_by_index_train(index)
        else:
            video_path, caption, video_id = \
                self._get_vidpath_and_caption_by_index_test(index)

        try:
            imgs, idxs = VideoCapture.load_frames(
                video_path, self.config.num_frames
            )
        except Exception:
            self.err += 1
            raise RuntimeError(f"Failed to load video: {video_path}")

        # --------------------------------------------------
        # image transforms
        # --------------------------------------------------
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        # --------------------------------------------------
        # category name -> category id
        # --------------------------------------------------
        category_name = self.video_to_category[video_id]
        category_id = self.category2id[category_name]

        # --------------------------------------------------
        # return dict (train & test unified)
        # --------------------------------------------------
        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
            'category': category_id,        # for loss / ETF / prototype
            'category_name': category_name  # for analysis / semantics
        }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)

    # ==================================================
    # helpers
    # ==================================================
    def _get_vidpath_and_caption_by_index_train(self, index):
        vid, caption = self.all_train_pairs[index]
        video_path = os.path.join(self.videos_train_dir, vid)
        return video_path, caption, vid

    def _get_vidpath_and_caption_by_index_test(self, index):
        vid, caption = self.all_test_pairs[index]
        video_path = os.path.join(self.videos_val_dir, vid)
        return video_path, caption, vid

    def _construct_all_train_pairs(self, benchmark):
        self.all_train_pairs = []
        for vid in self.videos:
            if benchmark == 'anet_clip':
                caption = self.vid2caption[vid]['sentences'][0]
                vid_clip = vid + '_1'
                self.all_train_pairs.append([vid_clip, caption])

            elif benchmark == 'anet_para':
                paragraph = ''
                for cap in self.vid2caption[vid]:
                    paragraph += cap + ' '
                self.all_train_pairs.append([vid, paragraph])

            elif benchmark == 'anet_cap':
                for cap in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, cap])

    def _construct_all_test_pairs(self, benchmark):
        self.all_test_pairs = []
        for vid in self.videos:
            if benchmark == 'anet_clip':
                caption = self.vid2caption[vid]['sentences'][0]
                vid_clip = vid + '_1'
                self.all_test_pairs.append([vid_clip, caption])

            elif benchmark == 'anet_para':
                paragraph = ''
                for cap in self.vid2caption[vid]:
                    paragraph += cap + ' '
                self.all_test_pairs.append([vid, paragraph])

            elif benchmark == 'anet_cap':
                for cap in self.vid2caption[vid]:
                    self.all_test_pairs.append([vid, cap])
