(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-30643776"],{"0dfd":function(t,e,n){"use strict";n.d(e,"a",(function(){return a})),n.d(e,"b",(function(){return r}));var a=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"errPage-container"},[n("el-button",{staticClass:"pan-back-btn",attrs:{icon:"arrow-left"},on:{click:t.back}},[t._v(" 返回 ")]),n("el-row",[n("el-col",{attrs:{span:12}},[n("h1",{staticClass:"text-jumbo text-ginormous"},[t._v(" 401错误! ")]),n("h2",[t._v("您没有访问权限！")]),n("h6",[t._v("对不起，您没有访问权限，请不要进行非法操作！您可以返回主页面")]),n("ul",{staticClass:"list-unstyled"},[n("li",{staticClass:"link-type"},[n("router-link",{attrs:{to:"/"}},[t._v(" 回首页 ")])],1)])]),n("el-col",{attrs:{span:12}},[n("img",{attrs:{src:t.errGif,width:"313",height:"428",alt:"Girl has dropped her ice cream."}})])],1)],1)},r=[]},4252:function(t,e,n){"use strict";n("4f12")},"4f12":function(t,e,n){},7022:function(t,e,n){"use strict";var a=n("4ea4").default;Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0;var r=a(n("cc6c")),i={name:"Page401",data:function(){return{errGif:r.default+"?"+ +new Date}},methods:{back:function(){this.$route.query.noGoBack?this.$router.push({path:"/"}):this.$router.go(-1)}}};e.default=i},cc6c:function(t,e,n){t.exports=n.p+"static/img/401.089007e7.gif"},da36:function(t,e,n){"use strict";n.r(e);var a=n("7022"),r=n.n(a);for(var i in a)["default"].indexOf(i)<0&&function(t){n.d(e,t,(function(){return a[t]}))}(i);e["default"]=r.a},ec55:function(t,e,n){"use strict";n.r(e);var a=n("0dfd"),r=n("da36");for(var i in r)["default"].indexOf(i)<0&&function(t){n.d(e,t,(function(){return r[t]}))}(i);n("4252");var c=n("2877"),u=Object(c["a"])(r["default"],a["a"],a["b"],!1,null,"f2e02586",null);e["default"]=u.exports}}]);