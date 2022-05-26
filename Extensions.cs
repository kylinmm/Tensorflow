using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;

namespace Tensorflow
{
    public static class Extensions
    {
        /// <summary>
        /// NDArray和mat转化（不使用指针）
        /// </summary>
        /// <param name="nDArray"></param>
        /// <returns></returns>
        public static Mat ToMat(this NDArray nDArray)
        {
            return new Mat((int)nDArray.shape[0], (int)nDArray.shape[1], nDArray.GetMatType(), nDArray.ToArray());
        }

        /// <summary>
        /// 将NDArray中的类型转化为mat对应的类型
        /// </summary>
        /// <param name="nDArray"></param>
        /// <returns></returns>
        public static MatType GetMatType(this NDArray nDArray)
        {
            int channels = nDArray.ndim == 3 ? (int)nDArray.shape[2] : 1;
            return nDArray.GetDataType() switch
            {
                TF_DataType.TF_INT32 => channels == 1 ? MatType.CV_32SC1 :
                channels == 2 ? MatType.CV_32SC2 :
                throw new ArgumentException($"nDArray data type = {nDArray.GetDataType()} & channels = {channels} is not supported"),

                TF_DataType.TF_FLOAT => channels == 1 ? MatType.CV_32FC1 :
                throw new ArgumentException($"nDArray data type = {nDArray.GetDataType()} & channels = {channels} is not supported"),

                TF_DataType.TF_DOUBLE => channels == 1 ? MatType.CV_64FC1 :
                throw new ArgumentException($"nDArray data type = {nDArray.GetDataType()} & channels = {channels} is not supported"),

                TF_DataType.TF_UINT8 => channels == 1 ? MatType.CV_8UC1 :
                channels == 3 ? MatType.CV_8UC3 :
                channels == 4 ? MatType.CV_8UC4 :
                throw new ArgumentException($"nDArray data type = {nDArray.GetDataType()} & channels = {channels} is not supported"),

                _ => throw new ArgumentException($"nDArray data type = {nDArray.GetDataType()} is not supported")
            };

        }
    }
}
