# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        # if not isinstance(path, list):
        #     path = [path]

        # samples = {}
        # for annotation_file in path:
        #     tmp = load_dataset_file(annotation_file)
        #     for s in tmp:
        #         seq_id = s["name"]
        #         if seq_id in samples:
        #             assert samples[seq_id]["name"] == s["name"]
        #             assert samples[seq_id]["signer"] == s["signer"]
        #             assert samples[seq_id]["gloss"] == s["gloss"]
        #             assert samples[seq_id]["text"] == s["text"]
        #             samples[seq_id]["sign"] = torch.cat(
        #                 [samples[seq_id]["sign"], s["sign"]], axis=1
        #             )
        #         else:
        #             samples[seq_id] = {
        #                 "name": s["name"],
        #                 "signer": s["signer"],
        #                 "gloss": s["gloss"],
        #                 "text": s["text"],
        #                 "sign": s["sign"],
        #             }
        # print(samples)
        '''
        print(samples)
        exit()
        {..., 'train/27January_2013_Sunday_tagesschau-8841': {'name': 'train/27January_2013_Sunday_tagesschau-8841', 'signer': 'Signer01', 'gloss': 'DONNERSTAG FREUNDLICH SONNE DANN SPAETER KOMMEN REGEN', 'text': 'der donnerstag beginnt oft freundlich später zieht von westen regen heran .', 'sign': tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.8989, 0.0000],
        [0.0000, 0.2920, 0.2877,  ..., 0.0000, 0.8531, 0.0000],
        [0.0215, 0.5436, 1.0011,  ..., 0.0000, 0.8510, 0.0000],
        ...,
        [0.0000, 0.9392, 0.0000,  ..., 0.0000, 0.9980, 0.0000],
        [0.0000, 1.2581, 0.0000,  ..., 0.0000, 0.7922, 0.0000],
        [0.0000, 1.4715, 0.0000,  ..., 0.0000, 0.7854, 0.0000]])}, 'train/27January_2013_Sunday_tagesschau-8842': {'name': 'train/27January_2013_Sunday_tagesschau-8842', 'signer': 'Signer01', 'gloss': 'BLEIBEN WIND', 'text': 'es bleibt windig .', 'sign': tensor([[0.9408, 0.0751, 0.0000,  ..., 0.0000, 0.6217, 0.0000],
        [2.5872, 0.0000, 0.0000,  ..., 0.0795, 0.7281, 0.0000],
        [1.3273, 0.0000, 0.0411,  ..., 0.0000, 0.8729, 0.0000],
        ...,
        [0.4032, 1.1689, 0.7295,  ..., 0.0000, 1.1873, 0.0000],
        [0.0432, 1.4143, 0.5994,  ..., 0.0000, 1.1695, 0.0000],
        [0.1225, 1.3719, 0.2212,  ..., 0.0000, 1.0340, 0.0000]])}}
        '''
        data_path = f'{path}'
        print(data_path, end = ': ')
        samples = {}
        f_files = open(data_path + '.files', "r")
        f_gloss = open(data_path + '.gloss', "r")
        f_skels = open(data_path + '.skels', "r")
        f_text = open(data_path + '.text', "r")
        files_list = f_files.readlines()
        gloss_list = f_gloss.readlines()
        skels_list = f_skels.readlines()
        text_list = f_text.readlines()
        print(len(files_list))
        for i in range(len(files_list)):
            samples[files_list[i].strip('\n')] = {
                "name": files_list[i].strip('\n'),
                "signer": 'Signer01',
                "gloss": gloss_list[i].strip('\n'),
                "text": text_list[i].strip('\n'),
                "sign": torch.tensor(eval(skels_list[i].strip('\n').replace(' ', ','))).reshape(-1, 151),
            }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
