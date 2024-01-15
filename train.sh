

RUN_MULTI_GPUS(){
accelerate config default

CUDA_VISIBLE_DEVICES=0,1 accelerate launch  --multi_gpu train_mnist.py
}

RUN_SINGLE_GPUS(){
accelerate config default

CUDA_VISIBLE_DEVICES=0 accelerate launch  train_mnist.py

}



RUN_MULTI_GPUS

