V3
train.txt = 3.001.001 + office bg (15)
train_5.txt = train.txt - coco-too-small
train_6.txt = train_5.txt + office bg (30)
train_7.txt = 3.001.001 + office bg (30)

(For Normal:)
train_11 = 3.001.001(530) + office bg (15) = 545
(For ppl cnt:)
train_12_1 = train_11 + normal(7) + pplcnt(5) = 557
train_12_2 = train_11 + normal(7) + pplcnt(10) = 562
train_12_3 = train_11 + normal(7) + pplcnt(13) = 565 
(For Half body & dist limit)
train_13_1 = train_12_3 + half body(5) = 570
train_13_2 = train_13_1 + bg (5) = 575
train_13_3 = train_13_2 + far(10) = 585
(For super low light)
train_14_1 = train_13_3 + lowlight ppl(5) = 590
train_14_2 = train_14_2 + lowlight bg(3) = 593

V4 
train_2 = 617 (taipei env)
train_3 = 624 (taipei bright env)