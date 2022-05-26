using System;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Optimizers;
using Tensorflow.NumPy;
using Tensorflow.Keras.Datasets;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using OpenCvSharp;

namespace Tensorflow
{
    public  class Gan
    {
        private  int BATCH_SIZE= 256;
        private  int BUFFER_SIZE= 60000;
        private  int noise_dim = 100;
        private string imgpath = "gan\\Imgs";
        private string modelpath = "gan\\Models";

        /// <summary>
        /// 初始化
        /// </summary>
        /// <param name="_BATCH_SIZE"></param>
        /// <param name="_BUFFER_SIZE"></param>
        /// <param name="_noise_dim">多少个随机数生成图片</param>
        public Gan(int _BATCH_SIZE,int _BUFFER_SIZE,int _noise_dim) {

             BATCH_SIZE = _BATCH_SIZE;
             BUFFER_SIZE = _BUFFER_SIZE;
             noise_dim = _noise_dim;

        }

        /// <summary>
        /// 导入keras手写数字数据集，并做归一化处理
        /// </summary>
        /// <returns></returns>
        public IDatasetV2 LoadBatchData()
        {        

            var ((train_images, train_labels), (_, _)) = keras.datasets.mnist.load_data();
            train_images = train_images.reshape(newshape: (train_images.shape[0],28,28,1)).astype(TF_DataType.TF_FLOAT);
            train_images = (train_images - 127.5) / 127.5;

            var datasets = tf.data.Dataset.from_tensor_slices(train_images);
                datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE);

            return datasets;
        }

        /// <summary>
        /// 建立生成器
        /// </summary>
        /// <returns></returns>
        public Model generator_model()
        {   // 用100个随机数（噪音）生成手写数据集
            var layers = new LayersApi();

            var model = keras.Sequential();
            model.add(layers.Dense(256, input_shape: 100, use_bias: false));
            model.add(layers.BatchNormalization());
            model.add(layers.LeakyReLU());


            model.add(layers.Dense(512, use_bias: false));
            model.add(layers.BatchNormalization());
            model.add(layers.LeakyReLU());

            model.add(layers.Dense(28 * 28 * 1, use_bias: false, activation: KerasApi.keras.activations.Tanh));
            model.add(layers.BatchNormalization());

            model.add(layers.Reshape((28,28,1)));

            return model;
        }

        /// <summary>
        /// 建立判别器
        /// </summary>
        /// <returns></returns>
        public Model discriminator_model()
        {   //识别输入的图片
            var layers = new LayersApi();
            var model = keras.Sequential();
            model.add(layers.Flatten());

            model.add(layers.Dense(512, use_bias : false));
            model.add(layers.BatchNormalization());
            model.add(layers.LeakyReLU());

            model.add(layers.Dense(256, use_bias : false));
            model.add(layers.BatchNormalization());
            model.add(layers.LeakyReLU());


            model.add(layers.Dense(1));


            return model;
        }

        /// <summary>
        /// 定义判别器损失函数
        /// </summary>
        /// <param name="real_out"></param>
        /// <param name="fake_out"></param>
        /// <returns></returns>
        public Tensor discriminator_loss(Tensor real_out, Tensor fake_out)
        {
            var real_loss = BinaryCrossentropy(tf.ones_like(real_out), real_out);
            var fake_loss = BinaryCrossentropy(tf.zeros_like(fake_out), fake_out);
            return real_loss + fake_loss;
        }

        /// <summary>
        /// 定义生成器损失函数
        /// </summary>
        /// <param name="fake_out"></param>
        /// <returns></returns>
        public Tensor generator_loss(Tensor fake_out)
        {
            return BinaryCrossentropy(tf.ones_like(fake_out), fake_out);
        }

        /// <summary>
        /// 定义训练过程
        /// </summary>
        /// <param name="images"></param>
        public void train_step(Tensors images,ref OptimizerV2 generator_opt, ref OptimizerV2 discriminator_opt,ref Model generator, ref Model discriminator) {
            

            var noise = tf.random.normal((BATCH_SIZE, noise_dim));
            var real_out = discriminator.Apply(images);

            var gen_image = generator.Apply(noise);
            var fake_out = discriminator.Apply(gen_image);

            var gen_loss = generator_loss(fake_out);
            var disc_loss = discriminator_loss(real_out, fake_out);

            using (var tape = tf.GradientTape(true))
            {
                var gradient_gen = tape.gradient(gen_loss, generator.trainable_variables);
                var gradient_disc = tape.gradient(disc_loss, discriminator.trainable_variables);


                var a = generator.trainable_variables.Select(x => x as ResourceVariable);
                var test = zip(gradient_gen, generator.trainable_variables.Select(x => x as ResourceVariable));

                generator_opt.apply_gradients(zip(gradient_gen, generator.trainable_variables.Select(x => x as ResourceVariable)));
                discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables.Select(x => x as ResourceVariable)));

            }

          
        }

        /// <summary>
        /// 训练主方法
        /// </summary>
        /// <param name="dataset">训练数据集</param>
        /// <param name="epochs">训练次数</param>
        public void train(IDatasetV2 dataset,int epochs)
        {
            System.IO.Directory.CreateDirectory(imgpath);
            System.IO.Directory.CreateDirectory(modelpath);

            var generator_opt = keras.optimizers.Adam(0.0001f);
            var discriminator_opt = keras.optimizers.Adam(0.0001f);

            var generator = generator_model();
            var discriminator = discriminator_model();

            for (int i=0; i <= epochs; i++) {

                foreach (var image_batch in dataset)
                {
                    train_step(image_batch.Item1,ref generator_opt,ref discriminator_opt,ref generator,ref discriminator);
                }            
            
            }

            //保存训练的模型
            generator.save_weights($"{modelpath}\\Model_{DateTime.Now.Day}_g.weights");
            discriminator.save_weights($"{modelpath}\\Model_{DateTime.Now.Day}_d.weights");

        }

        /// <summary>
        /// 测试训练好的模型并输出其生成的图片文件
        /// </summary>
        public void TestModel() 
        {
            var num_exp_to_generate = 16; // 生成16张图片
           var seed = tf.random.normal((num_exp_to_generate, noise_dim));  // 16组随机数组，每组含100个随机数，用来生成16张图片。


            var gan = generator_model();
            gan.load_weights($"{modelpath}\\Model_{DateTime.Now.Day}_g.weights");
            var tensor_result = gan.predict(seed);
            
            for(int i=0;i< tensor_result.Length; i++)
            {
                var gen_imgs = tensor_result[i].numpy();
                SaveImage(gen_imgs, i);
            }
           
        }

        /// <summary>
        /// 保存生成器生成的图片
        /// </summary>
        /// <param name="gen_imgs"></param>
        /// <param name="step"></param>
        private void SaveImage(NDArray gen_imgs, int step)
        {
            int img_rows = 28;
            int img_cols = 28;

            gen_imgs = gen_imgs * 0.5 + 0.5;
            var c = 5;
            var r = gen_imgs.shape[0] / c;
            var nDArray = np.zeros((img_rows * r, img_cols * c), dtype: np.float32);
            for (var i = 0; i < r; i++)
            {
                for (var j = 0; j < c; j++)
                {
                    var x = new Slice(i * img_rows, (i + 1) * img_cols);
                    var y = new Slice(j * img_rows, (j + 1) * img_cols);
                    var v = gen_imgs[i * r + j].reshape((img_rows, img_cols));
                    nDArray[x, y] = v;
                }
            }
            var t = nDArray.reshape((img_rows * r, img_cols * c)) * 255;
            Cv2.ImWrite($"{imgpath}\\{step}.png", gen_imgs.ToMat());
        }



        /// <summary>
        /// 二分类损失函数
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        private Tensor BinaryCrossentropy(Tensor x, Tensor y)
        {
            var shape = tf.reduce_prod(tf.shape(x));
            var count = tf.cast(shape, TF_DataType.TF_FLOAT);
            x = tf.clip_by_value(x, 1e-6f, 1.0f - 1e-6f);
            var z = y * tf.log(x) + (1 - y) * tf.log(1 - x);
            var result = -1.0f / count * tf.reduce_sum(z);
            return result;
        }



    }
}
