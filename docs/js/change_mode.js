
var global_sun_src;
var global_moon_src;

function changeToLight(body, markdown_part, box, change_article_boxes) {
    body.className = 'light';
    markdown_part.className = 'markdown-body markdown-light'
    box.src = global_sun_src;
    for (b of change_article_boxes) {
        b.classList.remove('change-dark');
    }
    var dirTree = document.querySelector('.dir-tree');
    dirTree.style.background = '#f6f8fa';
    var allLinks = dirTree.querySelectorAll('a');
    for (var i = 0; i < allLinks.length; i++) {
        allLinks[i].style.color = 'black';
    }

    var navigator = document.querySelector('.header-navigator');
    var allLinks = navigator.querySelectorAll('a');
    for (var i = 0; i < allLinks.length; i++) {
        allLinks[i].style.color = 'black';
    }

    var activate_links = dirTree.querySelectorAll('.link-active-dark');
    for (var activate_link of activate_links) {
        activate_link.className = 'link-active';
    }
    var search_bar = document.querySelector('.search-bar');
    if (search_bar) {
        search_bar.style.background = '#f6f8fa';
    }
    box.state = !box.state;
    localStorage.setItem('theme', 'light');
}

function changeToDark(body, markdown_part, box, change_article_boxes) {
    body.className = 'dark';
    markdown_part.className = 'markdown-body markdown-dark'
    box.src = global_moon_src;
    for (b of change_article_boxes) {
        b.classList.add('change-dark');
    }
    var dirTree = document.querySelector('.dir-tree');
    dirTree.style.background = '#252D38';
    var allLinks = dirTree.querySelectorAll('a');
    for (var i = 0; i < allLinks.length; i++) {
        allLinks[i].style.color = 'white';
    }

    var navigator = document.querySelector('.header-navigator');
    var allLinks = navigator.querySelectorAll('a');
    for (var i = 0; i < allLinks.length; i++) {
        allLinks[i].style.color = 'white';
    }

    var activate_links = dirTree.querySelectorAll('.link-active');
    for (var activate_link of activate_links) {
        activate_link.className = 'link-active-dark';
    }
    var search_bar = document.querySelector('.search-bar');
    if (search_bar) {
        search_bar.style.background = '#252D38';
    }
    box.state = !box.state;
    localStorage.setItem('theme', 'dark');
}

function changeThemeMode() {
    let body = document.body;
    let markdown_part = document.querySelector('.markdown-body')
    let box = document.getElementById('changeThemeMode')
    let change_article_boxes = document.getElementsByClassName('change-article')
    if (box.state) {
        changeToLight(body, markdown_part, box, change_article_boxes)
    } else {
        changeToDark(body, markdown_part, box, change_article_boxes)
    }

}

// 添加切换颜色
function addChangeModeButton(sun_src, moon_src) {
    global_sun_src = sun_src;
    global_moon_src = moon_src;
    var change_mode_button = document.createElement('img')
    change_mode_button.src = sun_src;
    change_mode_button.className = 'changeMode'
    change_mode_button.id = 'changeThemeMode'
    change_mode_button.onclick = changeThemeMode
    change_mode_button.state = false; // light: false | dark: true
    document.body.appendChild(change_mode_button)
    // 主题保持
    const savedTheme = localStorage.getItem('theme');
    // 如果保存的主题存在,则设置当前主题为保存的主题
    if (savedTheme) {
        let body = document.body;
        let markdown_part = document.querySelector('.markdown-body')
        let change_article_boxes = document.getElementsByClassName('change-article')
        if (savedTheme == "dark") {
            changeToDark(body, markdown_part, change_mode_button, change_article_boxes);
        } else {
            change_mode_button.state = true;
            changeToLight(body, markdown_part, change_mode_button, change_article_boxes)
        }
    }
}

