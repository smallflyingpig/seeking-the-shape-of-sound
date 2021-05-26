import sys
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from sklearn import metrics
from dataloaders.mm_dataset import AttackMMDataset
from base_container import BaseContainer

global_grads = {}
def save_grad(v):
    global global_grads
    def hook(grad):
        global_grads[v] = grad
    return hook

def get_grad(v):
    return global_grads.get(v, None)

def reset_grad(v):
    global global_grads
    global_grads[v] = None

def get_dict_str(d):
    s = ','.join(["{}:{:5.3f}".format(k,v) for k,v in d.items()])
    return s

def array2tensor_img(sample):
    imgs = []
    filenames = []
    gender = []
    for i in range(len(sample)):
        imgs.append(sample[i]['image'])
        filenames.append(sample[i]['image_filename'])
        gender.append(sample[i]['gender'])

    imgs = torch.stack(imgs, 0)
    return imgs, filenames, gender

def array2tensor_aud(sample):
    auds = []
    filenames = []
    gender = []
    for i in range(len(sample)):
        auds.append(sample[i]['audio'])
        filenames.append(sample[i]['audio_filename'])
        gender.append(sample[i]['gender'])

    auds = torch.stack(auds, 0)
    return auds, filenames, gender


def cosine_distance(x, y):
    return (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(-1)

class NetworkAttacker(BaseContainer):
    def __init__(self):
        super().__init__()
        self.init_evaluation_container()
        self.test_set = AttackMMDataset(self.args.evaluation.dataset, split='test')

    def verify_baseline(self):
        print("Verifing baseline...")
        self.model.eval()
        dataloader = DataLoader(self.test_set, batch_size=64, shuffle=True, drop_last=True)
        scores = []
        targets = []
        for idx, batch in tqdm(enumerate(dataloader)):
            img, img2, label = batch['img'], batch['img2'], batch['label']
            img, img2, label = map(lambda x: x.cuda() for x in (img, img2, label))
            feat = self.model.F.model(img)
            feat2 = self.model.F.model(img2)
            score = cosine_distance(feat, feat2)
            scores.append(score.detach().cpu().numpy())
            targets.append(label.detach().cpu().numpy())
        scores = np.concatenate(scores, dim=0)
        targets = np.concatenate(targets, dim=0)
        fpr, tpr, T = metrics.roc_curve(targets, scores, pos_label=1)
        return metrics.auc(fpr, tpr)

    def verify_attack(self):
        print("Verifing attack...")
        self.model.eval()
        dataloader = DataLoader(self.test_set, batch_size=32, shuffle=True, drop_last=True)
        scores = []
        targets = []
        for idx, batch in tqdm(enumerate(dataloader)):
            img, img2, wav, wav2, label = map(lambda k: batch[k].cuda(), ('img', 'img2', 'wav', 'wav2', 'label'))
            with torch.no_grad():
                img = self.model.normalize(img)
                img2 = self.model.normalize(img2)
                target_img_feat = self.model.F(img)
                target_wav_feat = self.model.V(wav)
                paired_img_feat = self.model.F(img2)
                paired_wav_feat = self.model.V(wav2)

            pos_mask = (label==1).squeeze()
            neg_mask = (label==0).squeeze()

            img_wav_sim = cosine_distance(target_img_feat, target_wav_feat)
            img_wav_sim = img_wav_sim[neg_mask].mean()
            img_img_sim = cosine_distance(target_img_feat, paired_img_feat)[neg_mask].mean()

            print(f"img wav sim: {img_wav_sim.item():5.3f}, img img sim: {img_img_sim.item():5.3f}")
            # scores.append(score.detach().cpu().numpy())
            # targets.append(label.detach().cpu().numpy())
        # scores = np.concatenate(scores, dim=0)
        # targets = np.concatenate(targets, dim=0)
        # fpr, tpr, T = metrics.roc_curve(targets, scores, pos_label=1)
        # return metrics.auc(fpr, tpr)


def main():
    attacker = NetworkAttacker()
    attacker.verify_attack()

if __name__ == "__main__" :
    main()