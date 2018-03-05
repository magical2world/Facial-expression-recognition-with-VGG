import tensorflow as tf
import vgg16
import vgg19
import config
from utils import *
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

def main(args):
    train_data='expression/fer2013-train'
    test_data='expression/fer2013-test'

    train_img,train_label=load_image(train_data)
    test_img,test_label=load_image(test_data)

    images=tf.placeholder(tf.float32,[None,224,224,3])
    label=tf.placeholder(tf.float32,[None,7])
    if args.network_model=="vgg16":
        vgg=vgg16.Vgg16(args.fine_tuning)
    else:
        vgg=vgg19.Vgg19(args.fine_tuning)
    predict=vgg.build(images)
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=predict))
    #if fine_tuning='all',we will use GradientDescent to optimize all variables
    #if fine_tuning!='all',we will only use Adam to optimize the fully connected layers
    if args.fine_tuning=="all":
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cross_entropy)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
    correct=tf.equal(tf.argmax(tf.nn.softmax(predict),1),tf.argmax(label,1))
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(tf.all_variables())
        train=next_batch(train_img, train_label,32)
        for i in range(args.train_step):
            x_batch,y_batch=train.next()
            loss,_,acc,=sess.run([cross_entropy,optimizer,accuracy]
                     ,feed_dict={images:x_batch,label:y_batch})
            if i%10==0:
                saver.save(sess,'save_variables/vgg.module',global_step=i)
                print('number %d loss is %f'%(i,loss))
                print('number %d accuracy is %f'%(i,acc))
        test_accuracy=0
        test=next_batch(test_img,test_label,32)
        for j in range(100):
            x_batch,y_batch=test.next()
            acc=sess.run(accuracy,feed_dict={images:x_batch,label:y_batch})
            test_accuracy+=acc
        print('test accuracy is %f'%(test_accuracy/100))

if __name__=='__main__':
    args=config.get_args()
    main(args)