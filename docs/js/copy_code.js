
var global_before_copy_url;
var global_after_copy_url;

function add(block) {

    var clip_board = document.createElement('img');
    clip_board.id = 'code_copy';
    clip_board.src = global_before_copy_url;
    clip_board.onclick = function () {
        clip_board.src = global_after_copy_url;
        var range = document.createRange();
        range.selectNodeContents(block.firstChild);
        var selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        navigator.clipboard.writeText(block.firstChild.innerText);
    }
    block.appendChild(clip_board)
}



function horizon_wheel(event, block, maxScroll) {
    event.preventDefault();
    const scrollAmount = event.deltaY * 0.5;
    const imageElement = document.getElementById('code_copy');
    const computedStyle = window.getComputedStyle(imageElement);
    const currentRight = parseInt(computedStyle.getPropertyValue('right'));
    
    // 判断是否已滚动到左右边缘
    if (block.scrollLeft === 0 && scrollAmount < 0) {
        // 已经滚动到左边缘并且向左滚动，不做任何操作
    } else if (maxScroll-block.scrollLeft <= 50 && scrollAmount > 0) {
        // 已经滚动到右边缘并且向右滚动，不做任何操作
        // imageElement.style.right = `${currentRight - scrollAmount}px`;
    } else {
        imageElement.style.right = `${currentRight - scrollAmount}px`;
        // console.log(block.scrollLeft,maxScroll)
        block.scrollLeft += scrollAmount;
    }
}

function remove(block) {
    var clip_board = document.getElementById('code_copy')
    block.removeChild(clip_board)
}


function addCodeCopy(before_copy_url, after_copy_url) {
    global_before_copy_url = before_copy_url;
    global_after_copy_url = after_copy_url;
    // 为所有代码段添加可以复制的标记
    var code_blocks = document.getElementsByTagName('pre')
    for (var i = 0; i < code_blocks.length; i++) {
        const code_block = code_blocks[i];
        code_block.addEventListener("mouseenter", () => add(code_block));
        code_block.addEventListener("mouseleave", () => remove(code_block));
        if (code_block.scrollWidth > code_block.clientWidth) {
            // 如果有横向滚动，阻止页面默认的竖直滚动，并将滚动事件重定向到 <pre> 元素上
            const blockWidth = code_block.offsetWidth;
            const maxScroll = code_block.scrollWidth - blockWidth;
            code_block.addEventListener('wheel', (event) => horizon_wheel(event, code_block, maxScroll));
        }
    }
}