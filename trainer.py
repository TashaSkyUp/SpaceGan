import tensorflow as tf
import numpy as np
from multiprocessing import Process,Queue
#from multiprocessing.queues import  SimpleQueue as Queue

def epoch(LQ: Queue,lcls,batch_size =64):
    a="what the heck"
    fc_trans=lcls["fc_trans"]
    np_images=lcls["np_images"]
    X_test=lcls["X_test"]
    #tf=lcls["tf"]
    
    #config = tf.ConfigProto()


    with tf.dev

        a = fc_trans.fit(np_images.astype("float16"),
                           np_images.astype("float16"),
                           validation_data=(X_test,X_test),
                           validation_steps=1,
                           epochs=1,
                           batch_size=batch_size,
                           verbose=1)
    LQ.put(a)
print(__name__)
if __name__ == "__main__":
    Q = Queue()
    epoch_proc = Process(target=epoch,args=(Q,))
    epoch_proc.start()
    print(1)
    a=Q.get()
    print(2)
    epoch_proc.join()
    print("main")