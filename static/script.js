var cloth = document.querySelector('#cloth');
var reader = new FileReader();

function previewImage() {
    var input = document.getElementById("upload-input");
    var uploadedImage = document.getElementById("uploaded-image");
    uploadedImage.src = URL.createObjectURL(input.files[0]);
    console.log(uploadedImage.src)
  }
  
   function selectImage(selected) {
    var input = document.getElementById("cloth");
    var uploadedImage = document.getElementById("selected-image");
    uploadedImage.src = URL.createObjectURL(input.files[0]);
    console.log(uploadedImage.src)
  }

  async function displayResult() {
    const response = await fetch('/img_final');
    const processedImageUrl = await response.text();
    
    const processedImage = document.getElementById('processed-image');
    processedImage.src = processedImageUrl;
  }