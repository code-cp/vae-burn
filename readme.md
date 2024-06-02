<div align="center">

# VAE-Burn 🔥

### Implementation of the VAE in Rust 🦀 + [Burn 🔥](https://burn.dev/).

</div>

## Dataset 

I collected some images of characters in Detective Conan, then converted them to sketch using [Anime2Sketch](https://github.com/Mukosame/Anime2Sketch)

<div style="display: flex; justify-content: space-between;">
    <img src="assets/22926830-1-0.855017.jpg" alt="Image 5" style="width: 15%;" />
    <img src="assets/22925913-2-0.876512.jpg" alt="Image 1" style="width: 15%;" />
    <img src="assets/22926511-2-0.818353.jpg" alt="Image 2" style="width: 15%;" />
    <img src="assets/22926619-1-0.865486.jpg" alt="Image 3" style="width: 15%;" />
    <img src="assets/22926647-2-0.748802.jpg" alt="Image 4" style="width: 15%;" />
</div>

Please note that the customized dataset will not be made public due to copyright issues, this repo only works with MNIST dataset. 

## Results 

Target and reconstruction 

<div style="display: flex; justify-content: space-between;">
    <img src="assets/target.png" alt="Image 1" style="width: 30%;" />
    <img src="assets/reconstruction.png" alt="Image 5" style="width: 30%;" />
</div>

Sampling results 

<div style="display: flex; justify-content: space-between;">
    <img src="assets/result_mean-2_var0.4.png" alt="Image 1" style="width: 30%;" />
    <img src="assets/result.png" alt="Image 5" style="width: 30%;" />
</div>

As a comparison, here are some results from a VAE implemented in pytorch 

![img](assets/full_losses.png)

![img](assets/pytorch_results.png)

Sketch to image results from [windy AI](https://windybot.com/sketch-to-image-ai)

<div style="display: flex; justify-content: space-between;">
    <img src="assets/ran.jpg" alt="Image 1" style="width: 30%;" />
    <img src="assets/conan.jpg" alt="Image 5" style="width: 30%;" />
    <img src="assets/girl.jpg" alt="Image 5" style="width: 30%;" />
</div>