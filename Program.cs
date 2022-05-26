using OpenCvSharp;
using System;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace Tensorflow
{
    class Program
    {
        static void Main(string[] args)
        {


            //    Gan g = new Gan(256, 60000, 100);

            //  var dataset=  g.LoadBatchData();
            //    g.train(dataset,100);

            LinearRegression linearRegression = new LinearRegression();
            linearRegression.Run();
        }
    }
}
