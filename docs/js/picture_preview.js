

var images = document.getElementsByTagName('img');
var preview_image = document.createElement('img')
var black_overlay = document.createElement('div')
black_overlay.id = 'black_overlay'
preview_image.onclick = closePreview;
var preview = false;

function previewImage(image) {
    var url = image.getAttribute('src');
    if (!preview) {
        black_overlay.style.display = 'block';
        preview_image.className = 'preview-image'
        preview_image.src = url;
        preview = !preview;
        // document.body.style.overflow = 'hidden'
    }
}

function closePreview() {
    preview_image.className = '';
    preview_image.src = ''
    black_overlay.style.display = 'none';
    preview = !preview;
    // document.body.style.overflow = auto;
}


for (var i = 0; i < images.length; i++) {
    if (images[i].className == 'changeMode') continue;
    const image = images[i];
    image.addEventListener('click', () => previewImage(image));
}

document.body.appendChild(preview_image)
document.body.append(black_overlay)