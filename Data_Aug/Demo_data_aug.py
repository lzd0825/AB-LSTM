from data import *
import time

data_gen_args = dict(rotation_range = 90,width_shift_range = 0.2,height_shift_range = 0.2,horizontal_flip=True,zoom_range = 0.3,fill_mode='nearest')

myGenerator = trainGenerator(2,'./Aug_example','train','train_gt',data_gen_args,save_to_dir = "./Aug_example")

print "myGenerator is >>> ", myGenerator

num_batch = 2
for i,batch in enumerate(myGenerator):
    print i
    if(i >= num_batch):
        break

print time.asctime( time.localtime(time.time()) )
print "All Done"