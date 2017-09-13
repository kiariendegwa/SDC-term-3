import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
   :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #load graph
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    init = tf.truncated_normal_initializer(stddev = 0.01)

    layer_7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding = 'same', kernel_initializer = init)
    layer_4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding = 'same', kernel_initializer = init)
    layer_3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding = 'same', kernel_initializer = init)

    upsample1 = tf.layers.conv2d_transpose(layer_7_1x1, num_classes, kernel_size=5, strides=2, padding = 'same', kernel_initializer = init)
    layer1 = tf.layers.batch_normalization(upsample1)  
    layer1 = tf.add(layer1, layer_4_1x1)

    upsample2 = tf.layers.conv2d_transpose(layer1, num_classes, kernel_size =5, strides=2, padding = 'same', kernel_initializer = init)
    layer2 = tf.layers.batch_normalization(upsample2)      
    layer2 = tf.add(layer2, layer_3_1x1)
    output = tf.layers.conv2d_transpose(layer2, num_classes, kernel_size=14, strides=8, padding = 'same', kernel_initializer = init)

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = correct_label, logits = logits))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # Add tensorboard to this to visualize training intuitively
    tf.summary.scalar("loss", cross_entropy_loss)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    logs_path = '/tmp/tensorflow_logs/image_segmentation/'
    
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    display_epoch = 5
    learn_rate = 0.0001
    avg_cost = 0
    print("Begin training ... ")

    for epoch in range(epochs):
        count = 0
        for images, labels in get_batches_fn(batch_size):
                     #training goes here
                   _, c, summary = sess.run([train_op, cross_entropy_loss, merged_summary_op],
                            feed_dict={input_image: images,
                            correct_label: labels,
                            learning_rate: learn_rate,
                            keep_prob: 0.8})
                   # Write logs at every iteration
                   summary_writer.add_summary(summary, epoch * batch_size + count)
                   # Compute average loss
                   avg_cost += c/batch_size
                   count+=1
                   print(count)
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            # Save the variables to disk.
            #save_path = saver.save(sess, "/tmp/models/model.ckpt", global_step=epoch)
            #print("Model saved in file: %s" % save_path)
                      
    print("Optimization Finished!")
    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
                      
#tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    
    epochs = 1
    batch_size = 32

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    
    with tf.Session() as sess:
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32)

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        #saver = tf.train.Saver()
        
        # Train NN using the train_nn function
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        sess.run(tf.global_variables_initializer())   
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

if __name__ == '__main__':
    run()
