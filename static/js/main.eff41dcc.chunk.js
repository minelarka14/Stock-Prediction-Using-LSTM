(this.webpackJsonpmyreactapp=this.webpackJsonpmyreactapp||[]).push([[0],{13:function(e,t,a){},14:function(e,t,a){},15:function(e,t,a){"use strict";a.r(t);var n=a(0),i=a.n(n),o=a(3),r=a.n(o),s=(a(13),a(4)),l=a(5),c=a(1),m=a(7),h=a(6),d=(a(14),{navLink:{color:"white",fontWeight:"400",textDecoration:"inherit"},navBrand:{fontWeight:"bold",color:"white",fontSize:"20px",textDecoration:"inherit"},navLinkActive:{color:"white",fontWeight:"650",textDecoration:"inherit"},indexDiv:{paddingTop:"2%",position:"absolute",top:"10%",left:"5%",color:"white",maxWidth:"50%",fontSize:"20px",textAlign:"justify",fontFamily:"Montserrat"}}),u=function(e){Object(m.a)(a,e);var t=Object(h.a)(a);function a(e){var n;return Object(s.a)(this,a),(n=t.call(this,e)).state={page:0},n.handleIndexPage=n.handleIndexPage.bind(Object(c.a)(n)),n.handleContactPage=n.handleContactPage.bind(Object(c.a)(n)),n.handleLicencePage=n.handleLicencePage.bind(Object(c.a)(n)),n}return Object(l.a)(a,[{key:"handleIndexPage",value:function(e){this.setState({page:0})}},{key:"handleContactPage",value:function(e){this.setState({page:1})}},{key:"handleLicencePage",value:function(e){this.setState({page:2})}},{key:"render",value:function(e){return i.a.createElement("div",{className:"AppMain",style:{backgroundColor:"#24252a"}},i.a.createElement("div",{className:"bs4-setup"},i.a.createElement("link",{rel:"stylesheet",href:"https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css",integrity:"sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z",crossorigin:"anonymous"}),i.a.createElement("script",{src:"https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js",integrity:"sha384-B4gt1jrGC7Jh4AgTPSdUtOBvfO8shuf57BaghqFfPlYxofvL8/KUEfYiJOMMV+rV",crossorigin:"anonymous"}),i.a.createElement("script",{src:"https://code.jquery.com/jquery-3.5.1.slim.min.js",integrity:"sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj",crossorigin:"anonymous"}),i.a.createElement("script",{src:"https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js",integrity:"sha384-9/reFTGAW83EW2RDu2S0VKaIzap3H66lZH81PoYlFhbGU+6BZp6G7niu735Sk7lN",crossorigin:"anonymous"})),i.a.createElement("div",null,i.a.createElement("div",{className:"navbar navbar-expand-sm bg-dark"},i.a.createElement("ul",{className:"navbar-nav"},i.a.createElement("li",{onClick:this.handleIndexPage,className:"navbar-brand"},i.a.createElement("a",{href:"#",style:0==this.state.page?d.navLinkActive:d.navLink},"LSTMScouts")),i.a.createElement("li",{onClick:this.handleContactPage,className:"nav-link"},i.a.createElement("a",{href:"#",style:1==this.state.page?d.navLinkActive:d.navLink},"Contact us")),i.a.createElement("li",{onClick:this.handleLicencePage,className:"nav-link"},i.a.createElement("a",{href:"#",style:2==this.state.page?d.navLinkActive:d.navLink},"Licence")),i.a.createElement("li",{className:"nav-link"},i.a.createElement("a",{href:"../data/LSTMScouts.zip",download:!0,style:d.navLink},"Download")))),i.a.createElement("div",null,0==this.state.page?i.a.createElement("div",{style:d.indexDiv},i.a.createElement("link",{href:"https://fonts.googleapis.com/css2?family=Montserrat:wght@500&display=swap",rel:"stylesheet"}),i.a.createElement("p",null,"This is a machine learning algorithm made with tensorflow and python. This was made for our open theme badge for scouts. It uses long short term memory as the name implies and uses tkinter and matplotlib for the GUI and visualisation. "),i.a.createElement("br",null),i.a.createElement("p",null," We created this project using The open theme badge requirements:")," ",i.a.createElement("br",null),i.a.createElement("ul",null,i.a.createElement("li",null,"Decide to pursue any area of interest or skill he / she is interested in that broadly covers a certain theme or category such as sports, aeronautics, science, performing arts, etc. ",i.a.createElement("br",null)),i.a.createElement("br",null),i.a.createElement("li",null,"Identify an area that is not covered by the existing syllabus of the Scout Proficiency Badges and submit a write-up or proposal of the project he/ she would like to do to the National Scout Roundtable, with support from the Scout Leader"))):1==this.state.page?i.a.createElement("div",null,i.a.createElement("div",{className:"contact"},i.a.createElement("br",null),i.a.createElement("p",{style:d.indexDiv}," Email us at: ",i.a.createElement("br",null),i.a.createElement("br",null),"muhammadosaid06@gmail ",i.a.createElement("br",null),"chew_ming_hong_ethan@s2019.ssts.edu.sg"))):2==this.state.page?i.a.createElement("div",{class:"licencediv",style:d.indexDiv},i.a.createElement("p",{class:"licencetext"},"MIT License",i.a.createElement("br",null),i.a.createElement("br",null),"Copyright (c) 2020 Muhammad Osaid and Ethan Chew"),i.a.createElement("br",null),i.a.createElement("p",null,'Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:'),i.a.createElement("p",null,i.a.createElement("br",null),"The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software."),i.a.createElement("br",null),i.a.createElement("p",null,'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'),i.a.createElement("br",null),i.a.createElement("p",null,"Any opinions, chats, messages, news, research, analyses, prices, or other information contained on this project are provided as general market information for educational and entertainment purposes only, and do not constitute investment advice. The program should not be relied upon as a substitute for extensive independent market research before making your actual trading decisions. Opinions, market data, recommendations or any other content is subject to change at any time without notice. The contributers will not accept liability for any loss or damage, including without limitation any loss of profit, which may arise directly or indirectly from use of or reliance on such information."),i.a.createElement("br",null),i.a.createElement("br",null),i.a.createElement("br",null)):null)))}}]),a}(i.a.Component);Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));r.a.render(i.a.createElement(i.a.StrictMode,null,i.a.createElement(u,null)),document.getElementById("root")),document.body.style="background: #24252a;","serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()})).catch((function(e){console.error(e.message)}))},8:function(e,t,a){e.exports=a(15)}},[[8,1,2]]]);
//# sourceMappingURL=main.eff41dcc.chunk.js.map