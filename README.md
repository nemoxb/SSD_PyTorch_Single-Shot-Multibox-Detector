## Object Detection: SSD (Singel Shot Multibox Detector)
This model is based on a SSD architecture proposed in ECCV 2016 paper **SSD: Single Shot MultiBox Detector** by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.

## Quick Start
### Dependencies and Environment
- PyTorch >= 0.4.0
- Numpy >= 1.15
- Python >=3.5.0
- OpenCV
- Dataset [Download](https://drive.google.com/u/0/uc?id=197RBFt2niCcVNPmEUwzp3ds5LsWxZVxd&export=download) and extract it under default repository


### Train from scratch
```python
python main.py
```
you can specify training epochs, learning rate, batch size and if using augmentation. such as
```python
python main.py --epoch 300 --lr 0.01 --batch_size 16 --aug True
```


### Evaluate with Your checkpoints
```python
python main.py --test --epoch {SAVED_CHECKPOINT_NUMBER} --txt 3
```
--txt specifiy which dataset to test 1-train set 2-validation set 3-test set

### Results
cat **before** NMS

<img src="imgs/cat.jpg" width="300">

cat **after** NMS

<img src="imgs/cat_NMS.jpg" width="300">

dog **before** NMS

<img src="imgs/dog.jpg" width="300">

dog **after** NMS

<img src="imgs/dog_NMS.jpg" width="300">

one person **before** NMS

<img src="imgs/one_person.jpg" width="300">

one person **after** NMS

<img src="imgs/one_person_NMS.jpg" width="300">

two persons **before** NMS

<img src="imgs/two_persons.jpg" width="300">

two persons **after** NMS

<img src="imgs/two_persons_NMS.jpg" width="300">

