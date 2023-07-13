
var inputs = document.getElementsByTagName('input')
for(var i=0;i<inputs.length;i++) {
    inputs[i].removeAttribute('disabled')
    inputs[i].onclick = function() {
        return false;
    }
}

var markdown_part = document.querySelector(".markdown-body");
markdown_part.className = 'markdown-body markdown-light'

var currentUrl = window.location.href.slice(0, -1);
var dirTree = document.querySelector(".dir-tree");
var links = dirTree.querySelectorAll("a");
// 主题保持
const savedTheme = localStorage.getItem('theme');
// 如果保存的主题存在,则设置当前主题为保存的主题
links.forEach(function(link) {
  if (link.href === currentUrl) {
    link.scrollIntoView({block: 'center', inline:'nearest', container: dirTree });
    if (savedTheme) {
        if (savedTheme == 'dark') {
            link.classList.add("link-active-dark");
        } else {
            link.classList.add("link-active");
        }
    } else {
        link.classList.add("link-active");
    }
  }
});

