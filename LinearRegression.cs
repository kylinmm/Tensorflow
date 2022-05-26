using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.NumPy;


namespace Tensorflow
{
   public  class LinearRegression
    {
        public void Run()
        {

            // Supper Parameters
            float learning_rate = 0.01f;

            var W = tf.Variable<float>(1);
            var b = tf.Variable<float>(0);

            int epochs = 30;
            int steps = 100;
            Tensor loss = null;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int step = 0; step < steps; step++)
                {
                    int batch_size = 10;
                    (NDArray train_X, NDArray train_Y) = LoadBatchData(batch_size);

                    using (var g = tf.GradientTape())
                    {
                        //通过当前参数计算预测值
                        var pred_y = W * train_X + b;

                        //计算预测值和实际值的误差
                        loss = tf.reduce_sum(tf.pow(pred_y - train_Y, 2)) / batch_size;

                        //计算梯度
                        var gradients = g.gradient(loss, (W, b));

                        //更新参数
                        W.assign_sub(learning_rate * gradients.Item1);
                        b.assign_sub(learning_rate * gradients.Item2);
                    }
                }

                Console.WriteLine($"Epoch{epoch + 1}: loss = {loss.numpy()}; W={W.numpy()},b={b.numpy()}");
            }
        }

        public (NDArray, NDArray) LoadBatchData(int n_samples)
        {
            float w = 0.02f;
            float b = 1.0f;

            NDArray train_X = np.arange<float>(start: 1, end: n_samples + 1);
            NDArray train_Y = train_X * w + b;

            return (train_X, train_Y);
        }

    }
}
