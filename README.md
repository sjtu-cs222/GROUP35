Approximate Image Classifier
==
This is a project about approximate image classifier.
Data set
===
CIFAR-10. And it would be downloaded in the same menu of your python file if you don't have when you run python file
The whole Process
==
re-label
--
run /code/train/label1.py, it would create suppose_label.txt
train gate function
--
run code/train.py, it would cost a lot of time
validate on the test set
--
get into /code/MSDnet then run 
```
python main.py --model msdnet -b 20  -j 2 cifar10 --growth 6-12-24 --gpu 0 --resume --evaluate-from anytime_cifar_10.pth.tar
```
you would see the answers
I have upload my pre-trained gate function models and use it in main.py. So you can directly validate using command.
And if you want to use your own gate function models, edit code in line 468 in main.py
