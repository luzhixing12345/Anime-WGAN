
function addButton(x,text,url) {

    var button = document.createElement('button');
    button.innerText = text;
    button.setAttribute('url',url)
    button.className = 'change-article';
    button.onclick = function () {
        window.location= this.getAttribute('url')
    }
    x.appendChild(button)
}

function addLink(front_url,next_url,control) {
    
    let body = document.body;
    var next_front = document.createElement('div')
    next_front.className = 'next-front'

    // a: 只激活前一个
    // b: 只激活后一个
    // ab: 全部激活
    // x: 全部不激活
    if (control == 'x') {
        return;
    } else if (control == 'a') {
        addButton(next_front,'上一个',front_url)
    } else if (control == 'b') {
        addButton(next_front,'下一个',next_url)
    } else {
        addButton(next_front,'上一个',front_url)
        addButton(next_front,'下一个',next_url)
    }

    body.appendChild(next_front)
}

