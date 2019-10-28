import os
import time
import numpy as np
import tensorflow as tf
import pickle
from multiprocessing.pool import ThreadPool
import cv2

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()


def append_to_list(self, img):
    

    self.org_img_list.append(img)
    im = self.framework.resize_input(img)
    self.img_list.append(im)
    
def detection(self):
    
    inputs = np.zeros((len(self.img_list), 416, 416, 3), dtype='float64')

    for i in range(len(self.img_list)):
        inputs[i] = self.img_list[i]

    feed_dict = {self.inp : inputs}
    s_time = time.time()
    
    out = self.sess.run(self.out, feed_dict)
    threshold = self.FLAGS.threshold
    
    #print ('********* {} pictures '.format(len(inputs)))
    #print ('Inference time :: {} frames / {}'.format(len(inputs), time.time()-s_time))
    
    results = []
    result_img = []
    
    for i in range(len(out)):

        results.append([])
        boxes = self.framework.findboxes(out[i])

        org_img = self.org_img_list[i]


        for j in range(len(boxes)):
            tmpBox = self.framework.process_box(boxes[j], int(os.environ['HEIGHT']), int(os.environ['WIDTH']), threshold)
            if tmpBox is not None:
                
                x = tmpBox[0]
                y = tmpBox[2]
                w = tmpBox[1] - tmpBox[0]
                h = tmpBox[3] - tmpBox[2]
                
                detected_type = tmpBox[4]
                detected_prob = tmpBox[6]
                
 
                #cropped_img = org_img[y:y+h, x:x+w]
                #r, buf = cv2.imencode('.jpg', cropped_img)
                #obj['img'] = buf
                
                
                cv2.rectangle(org_img, (x, y + h), (x+ w, y), color=(255,255,255), thickness=3)    
                cv2.putText(org_img,  detected_type +' : '+ str(round(detected_prob, 2)), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color=(255,255,255), thickness=2)
                
        result_img.append(org_img)
                
    
    self.org_img_list = []
    self.img_list = []
    
    #print ('Inference + postprocess time :: {} frames / {}'.format(len(inputs), time.time()-s_time))
    
    
    return result_img

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op, self.summary_op] 
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)

        
        
        
        
        
def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)

    asdf = np.zeros((2, 608, 608, 3), dtype='float64')

    for i in range(len(asdf)):
        asdf[i] = im

    feed_dict = {self.inp : asdf}
    
    out = self.sess.run(self.out, feed_dict)
    threshold = self.FLAGS.threshold
    
    for i in range(len(out)):
        boxes = self.framework.findboxes(out[i])
        for box in boxes:
            tmpBox = self.framework.process_box(box, h, w, threshold)
            print (tmpBox)
        print ('---------------------------')
            
    
    
    '''
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        print (tmpBox)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

    '''


import math



def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
