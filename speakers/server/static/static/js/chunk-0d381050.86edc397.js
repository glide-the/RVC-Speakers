(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-0d381050"],{"0c47":function(t,e,r){var a=r("da84"),o=r("d44e");o(a.JSON,"JSON",!0)},"131a":function(t,e,r){var a=r("23e7"),o=r("d2bb");a({target:"Object",stat:!0},{setPrototypeOf:o})},"1e4b":function(t,e,r){"use strict";r.r(e);var a=r("c94b"),o=r("75e5");for(var n in o)["default"].indexOf(n)<0&&function(t){r.d(e,t,(function(){return o[t]}))}(n);r("60f4");var i=r("2877"),l=Object(i["a"])(o["default"],a["a"],a["b"],!1,null,"62fc0152",null);e["default"]=l.exports},"23dc":function(t,e,r){var a=r("d44e");a(Math,"Math",!0)},3410:function(t,e,r){var a=r("23e7"),o=r("d039"),n=r("7b0b"),i=r("e163"),l=r("e177"),c=o((function(){i(1)}));a({target:"Object",stat:!0,forced:c,sham:!l},{getPrototypeOf:function(t){return i(n(t))}})},"365c":function(t,e,r){"use strict";var a=r("4ea4").default;Object.defineProperty(e,"__esModule",{value:!0}),e.runnerSubmit=e.runnerResult=void 0;var o=a(r("e5a8")),n=function(t){return(0,o.default)({url:"/runner/submit",method:"post",data:t})};e.runnerSubmit=n;var i=function(t){return(0,o.default)({url:"/runner/result",method:"get",responseType:"blob",params:t})};e.runnerResult=i},"4dc3":function(t,e,r){"use strict";var a=r("4ea4").default;Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0,r("d3b7"),r("3ca3"),r("ddb0"),r("2b3d"),r("9861");var o=a(r("7ec2")),n=a(r("c973")),i=(a(r("bc3a")),r("365c"));function l(t,e,r){var a=t<e?1:-1,o=t+a*r;return 1===a&&o>e||-1===a&&o<e?e:o}var c={data:function(){return{formData:{created_at:0,requested_at:0,parameter:{task_name:"bark_voice_task"},payload:{bark:{text:"青春的时间转瞬即逝，秋水于海又岂有一只浮游",speaker_history_prompt:"zh_speaker_2",text_temp:1,waveform_temp:.9},rvc:{model_index:0,f0_up_key:5,f0_method:"rmvpe",index_rate:.9,filter_radius:1,rms_mix_rate:1,resample_sr:0,protect:.33,f0_file:""}}},showProgressBar:!1,progress:0,showAudioPlayer:!1,src:null,foldBarkParams:!0,foldRvcParams:!0}},methods:{submitForm:function(){var t=this;return(0,n.default)((0,o.default)().mark((function e(){var r;return(0,o.default)().wrap((function(e){while(1)switch(e.prev=e.next){case 0:return t.showProgressBar=!0,t.progress=0,e.next=4,(0,i.runnerSubmit)(t.formData);case 4:if(r=e.sent,200!==r.data.code||!0!==r.data.data.finished){e.next=11;break}return t.progress=100,e.next=9,t.getResultAndDisplayAudio(r.data.data.task_id);case 9:e.next=13;break;case 11:return e.next=13,t.pollTaskStatus(r.data.data.task_id);case 13:case"end":return e.stop()}}),e)})))()},pollTaskStatus:function(t){var e=this;return(0,n.default)((0,o.default)().mark((function r(){var a;return(0,o.default)().wrap((function(r){while(1)switch(r.prev=r.next){case 0:a=setInterval((0,n.default)((0,o.default)().mark((function r(){var n;return(0,o.default)().wrap((function(r){while(1)switch(r.prev=r.next){case 0:return e.progress=l(e.progress,100,20),r.next=3,(0,i.runnerSubmit)(e.formData);case 3:if(n=r.sent,200!==n.data.code||!0!==n.data.data.finished){r.next=9;break}return clearInterval(a),e.progress=100,r.next=9,e.getResultAndDisplayAudio(t);case 9:case"end":return r.stop()}}),r)}))),5e3);case 1:case"end":return r.stop()}}),r)})))()},getResultAndDisplayAudio:function(t){var e=this;return(0,n.default)((0,o.default)().mark((function r(){var a;return(0,o.default)().wrap((function(r){while(1)switch(r.prev=r.next){case 0:return r.prev=0,r.next=3,(0,i.runnerResult)({task_id:t});case 3:return a=r.sent,e.showAudioPlayer=!0,e.src=window.URL.createObjectURL(a.data),r.next=8,e.$refs.audio.play();case 8:r.next=12;break;case 10:r.prev=10,r.t0=r["catch"](0);case 12:case"end":return r.stop()}}),r,null,[[0,10]])})))()}}};e.default=c},"60f4":function(t,e,r){"use strict";r("b42f")},"75e5":function(t,e,r){"use strict";r.r(e);var a=r("4dc3"),o=r.n(a);for(var n in a)["default"].indexOf(n)<0&&function(t){r.d(e,t,(function(){return a[t]}))}(n);e["default"]=o.a},"7ec2":function(t,e,r){r("a4d3"),r("e01a"),r("d3b7"),r("d28b"),r("3ca3"),r("ddb0"),r("b636"),r("944a"),r("0c47"),r("23dc"),r("d9e2"),r("3410"),r("159b"),r("b0c0"),r("131a"),r("fb6a");var a=r("7037")["default"];function o(){"use strict";
/*! regenerator-runtime -- Copyright (c) 2014-present, Facebook, Inc. -- license (MIT): https://github.com/facebook/regenerator/blob/main/LICENSE */t.exports=o=function(){return e},t.exports.__esModule=!0,t.exports["default"]=t.exports;var e={},r=Object.prototype,n=r.hasOwnProperty,i="function"==typeof Symbol?Symbol:{},l=i.iterator||"@@iterator",c=i.asyncIterator||"@@asyncIterator",s=i.toStringTag||"@@toStringTag";function u(t,e,r){return Object.defineProperty(t,e,{value:r,enumerable:!0,configurable:!0,writable:!0}),t[e]}try{u({},"")}catch(E){u=function(t,e,r){return t[e]=r}}function f(t,e,r,a){var o=e&&e.prototype instanceof m?e:m,n=Object.create(o.prototype),i=new L(a||[]);return n._invoke=function(t,e,r){var a="suspendedStart";return function(o,n){if("executing"===a)throw new Error("Generator is already running");if("completed"===a){if("throw"===o)throw n;return j()}for(r.method=o,r.arg=n;;){var i=r.delegate;if(i){var l=k(i,r);if(l){if(l===d)continue;return l}}if("next"===r.method)r.sent=r._sent=r.arg;else if("throw"===r.method){if("suspendedStart"===a)throw a="completed",r.arg;r.dispatchException(r.arg)}else"return"===r.method&&r.abrupt("return",r.arg);a="executing";var c=p(t,e,r);if("normal"===c.type){if(a=r.done?"completed":"suspendedYield",c.arg===d)continue;return{value:c.arg,done:r.done}}"throw"===c.type&&(a="completed",r.method="throw",r.arg=c.arg)}}}(t,r,i),n}function p(t,e,r){try{return{type:"normal",arg:t.call(e,r)}}catch(E){return{type:"throw",arg:E}}}e.wrap=f;var d={};function m(){}function v(){}function h(){}var y={};u(y,l,(function(){return this}));var b=Object.getPrototypeOf,_=b&&b(b(O([])));_&&_!==r&&n.call(_,l)&&(y=_);var x=h.prototype=m.prototype=Object.create(y);function g(t){["next","throw","return"].forEach((function(e){u(t,e,(function(t){return this._invoke(e,t)}))}))}function w(t,e){function r(o,i,l,c){var s=p(t[o],t,i);if("throw"!==s.type){var u=s.arg,f=u.value;return f&&"object"==a(f)&&n.call(f,"__await")?e.resolve(f.__await).then((function(t){r("next",t,l,c)}),(function(t){r("throw",t,l,c)})):e.resolve(f).then((function(t){u.value=t,l(u)}),(function(t){return r("throw",t,l,c)}))}c(s.arg)}var o;this._invoke=function(t,a){function n(){return new e((function(e,o){r(t,a,e,o)}))}return o=o?o.then(n,n):n()}}function k(t,e){var r=t.iterator[e.method];if(void 0===r){if(e.delegate=null,"throw"===e.method){if(t.iterator["return"]&&(e.method="return",e.arg=void 0,k(t,e),"throw"===e.method))return d;e.method="throw",e.arg=new TypeError("The iterator does not provide a 'throw' method")}return d}var a=p(r,t.iterator,e.arg);if("throw"===a.type)return e.method="throw",e.arg=a.arg,e.delegate=null,d;var o=a.arg;return o?o.done?(e[t.resultName]=o.value,e.next=t.nextLoc,"return"!==e.method&&(e.method="next",e.arg=void 0),e.delegate=null,d):o:(e.method="throw",e.arg=new TypeError("iterator result is not an object"),e.delegate=null,d)}function D(t){var e={tryLoc:t[0]};1 in t&&(e.catchLoc=t[1]),2 in t&&(e.finallyLoc=t[2],e.afterLoc=t[3]),this.tryEntries.push(e)}function P(t){var e=t.completion||{};e.type="normal",delete e.arg,t.completion=e}function L(t){this.tryEntries=[{tryLoc:"root"}],t.forEach(D,this),this.reset(!0)}function O(t){if(t){var e=t[l];if(e)return e.call(t);if("function"==typeof t.next)return t;if(!isNaN(t.length)){var r=-1,a=function e(){for(;++r<t.length;)if(n.call(t,r))return e.value=t[r],e.done=!1,e;return e.value=void 0,e.done=!0,e};return a.next=a}}return{next:j}}function j(){return{value:void 0,done:!0}}return v.prototype=h,u(x,"constructor",h),u(h,"constructor",v),v.displayName=u(h,s,"GeneratorFunction"),e.isGeneratorFunction=function(t){var e="function"==typeof t&&t.constructor;return!!e&&(e===v||"GeneratorFunction"===(e.displayName||e.name))},e.mark=function(t){return Object.setPrototypeOf?Object.setPrototypeOf(t,h):(t.__proto__=h,u(t,s,"GeneratorFunction")),t.prototype=Object.create(x),t},e.awrap=function(t){return{__await:t}},g(w.prototype),u(w.prototype,c,(function(){return this})),e.AsyncIterator=w,e.async=function(t,r,a,o,n){void 0===n&&(n=Promise);var i=new w(f(t,r,a,o),n);return e.isGeneratorFunction(r)?i:i.next().then((function(t){return t.done?t.value:i.next()}))},g(x),u(x,s,"Generator"),u(x,l,(function(){return this})),u(x,"toString",(function(){return"[object Generator]"})),e.keys=function(t){var e=[];for(var r in t)e.push(r);return e.reverse(),function r(){for(;e.length;){var a=e.pop();if(a in t)return r.value=a,r.done=!1,r}return r.done=!0,r}},e.values=O,L.prototype={constructor:L,reset:function(t){if(this.prev=0,this.next=0,this.sent=this._sent=void 0,this.done=!1,this.delegate=null,this.method="next",this.arg=void 0,this.tryEntries.forEach(P),!t)for(var e in this)"t"===e.charAt(0)&&n.call(this,e)&&!isNaN(+e.slice(1))&&(this[e]=void 0)},stop:function(){this.done=!0;var t=this.tryEntries[0].completion;if("throw"===t.type)throw t.arg;return this.rval},dispatchException:function(t){if(this.done)throw t;var e=this;function r(r,a){return i.type="throw",i.arg=t,e.next=r,a&&(e.method="next",e.arg=void 0),!!a}for(var a=this.tryEntries.length-1;a>=0;--a){var o=this.tryEntries[a],i=o.completion;if("root"===o.tryLoc)return r("end");if(o.tryLoc<=this.prev){var l=n.call(o,"catchLoc"),c=n.call(o,"finallyLoc");if(l&&c){if(this.prev<o.catchLoc)return r(o.catchLoc,!0);if(this.prev<o.finallyLoc)return r(o.finallyLoc)}else if(l){if(this.prev<o.catchLoc)return r(o.catchLoc,!0)}else{if(!c)throw new Error("try statement without catch or finally");if(this.prev<o.finallyLoc)return r(o.finallyLoc)}}}},abrupt:function(t,e){for(var r=this.tryEntries.length-1;r>=0;--r){var a=this.tryEntries[r];if(a.tryLoc<=this.prev&&n.call(a,"finallyLoc")&&this.prev<a.finallyLoc){var o=a;break}}o&&("break"===t||"continue"===t)&&o.tryLoc<=e&&e<=o.finallyLoc&&(o=null);var i=o?o.completion:{};return i.type=t,i.arg=e,o?(this.method="next",this.next=o.finallyLoc,d):this.complete(i)},complete:function(t,e){if("throw"===t.type)throw t.arg;return"break"===t.type||"continue"===t.type?this.next=t.arg:"return"===t.type?(this.rval=this.arg=t.arg,this.method="return",this.next="end"):"normal"===t.type&&e&&(this.next=e),d},finish:function(t){for(var e=this.tryEntries.length-1;e>=0;--e){var r=this.tryEntries[e];if(r.finallyLoc===t)return this.complete(r.completion,r.afterLoc),P(r),d}},catch:function(t){for(var e=this.tryEntries.length-1;e>=0;--e){var r=this.tryEntries[e];if(r.tryLoc===t){var a=r.completion;if("throw"===a.type){var o=a.arg;P(r)}return o}}throw new Error("illegal catch attempt")},delegateYield:function(t,e,r){return this.delegate={iterator:O(t),resultName:e,nextLoc:r},"next"===this.method&&(this.arg=void 0),d}},e}t.exports=o,t.exports.__esModule=!0,t.exports["default"]=t.exports},"944a":function(t,e,r){var a=r("d066"),o=r("e065"),n=r("d44e");o("toStringTag"),n(a("Symbol"),"Symbol")},b42f:function(t,e,r){},b636:function(t,e,r){var a=r("e065");a("asyncIterator")},c94b:function(t,e,r){"use strict";r.d(e,"a",(function(){return a})),r.d(e,"b",(function(){return o}));var a=function(){var t=this,e=t.$createElement,r=t._self._c||e;return r("div",[r("el-container",[r("el-header",[r("h1",[t._v("演示页面")])]),r("el-footer",[r("a",{attrs:{href:"/docs"}},[t._v("查看文档")]),r("a",{attrs:{target:"_blank",href:"https://github.com/glide-the/RVC-Speakers"}},[t._v("项目地址")])]),r("el-footer",{attrs:{title:"录音播放",visible:t.showAudioPlayer},on:{"update:visible":function(e){t.showAudioPlayer=e}}},[r("audio",{ref:"audio",attrs:{src:t.src,autoplay:"autoplay",controls:"controls"}},[t._v(" Your browser does not support the audio element. ")])]),r("el-main",[r("el-button",{attrs:{type:"primary"},on:{click:t.submitForm}},[t._v("提交")]),t.showProgressBar?r("el-progress",{attrs:{percentage:t.progress}}):t._e(),r("el-form",{ref:"form",attrs:{model:t.formData,"label-width":"150px"}},[r("el-row",[r("el-col",{attrs:{span:12}},[r("el-card",[r("div",{attrs:{slot:"header"},slot:"header"},[t._v(" BarkProcessorData Parameters "),r("div",{staticStyle:{"text-align":"right","margin-top":"10px"}},[r("el-button",{attrs:{type:"text"},on:{click:function(e){t.foldBarkParams=!t.foldBarkParams}}},[t._v(" "+t._s(t.foldBarkParams?"展开":"折叠")+" ")])],1)]),t.foldBarkParams?t._e():r("div",[r("el-form-item",{attrs:{label:"text"}},[r("el-tooltip",{attrs:{content:"生成文本",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.bark.text,callback:function(e){t.$set(t.formData.payload.bark,"text",e)},expression:"formData.payload.bark.text"}})],1)],1),r("el-form-item",{attrs:{label:"speaker_history_prompt"}},[r("el-tooltip",{attrs:{content:"音频预设npz文件",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.bark.speaker_history_prompt,callback:function(e){t.$set(t.formData.payload.bark,"speaker_history_prompt",e)},expression:"formData.payload.bark.speaker_history_prompt"}})],1)],1),r("el-form-item",{attrs:{label:"text_temp"}},[r("el-tooltip",{attrs:{content:"提示特殊标记程序，趋近于1，提示词特殊标记越明显",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.bark.text_temp,callback:function(e){t.$set(t.formData.payload.bark,"text_temp",e)},expression:"formData.payload.bark.text_temp"}})],1)],1),r("el-form-item",{attrs:{label:"waveform_temp"}},[r("el-tooltip",{attrs:{content:"提示隐藏空间转音频参数比例",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.bark.waveform_temp,callback:function(e){t.$set(t.formData.payload.bark,"waveform_temp",e)},expression:"formData.payload.bark.waveform_temp"}})],1)],1)],1)])],1),r("el-col",{attrs:{span:12}},[r("el-card",[r("div",{attrs:{slot:"header"},slot:"header"},[t._v(" RvcProcessorData Parameters "),r("div",{staticStyle:{"text-align":"right","margin-top":"10px"}},[r("el-button",{attrs:{type:"text"},on:{click:function(e){t.foldRvcParams=!t.foldRvcParams}}},[t._v(" "+t._s(t.foldRvcParams?"展开":"折叠")+" ")])],1)]),t.foldRvcParams?t._e():r("div",[r("el-form-item",{attrs:{label:" (f0_up_key)"}},[r("el-tooltip",{attrs:{content:"变调",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.rvc.f0_up_key,callback:function(e){t.$set(t.formData.payload.rvc,"f0_up_key",e)},expression:"formData.payload.rvc.f0_up_key"}})],1)],1),r("el-form-item",{attrs:{label:" (f0_file, 可选)"}},[r("el-tooltip",{attrs:{content:"F0曲线文件",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.rvc.f0_file,callback:function(e){t.$set(t.formData.payload.rvc,"f0_file",e)},expression:"formData.payload.rvc.f0_file"}})],1)],1),r("el-form-item",{attrs:{label:" (protect)"}},[r("el-tooltip",{attrs:{content:"保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.rvc.protect,callback:function(e){t.$set(t.formData.payload.rvc,"protect",e)},expression:"formData.payload.rvc.protect"}})],1)],1),r("el-form-item",{attrs:{label:"model_index"}},[r("el-tooltip",{attrs:{content:"模型索引",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.rvc.model_index,callback:function(e){t.$set(t.formData.payload.rvc,"model_index",e)},expression:"formData.payload.rvc.model_index"}})],1)],1),r("el-form-item",{attrs:{label:" (f0_method)"}},[r("el-tooltip",{attrs:{content:"F0方法",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.rvc.f0_method,callback:function(e){t.$set(t.formData.payload.rvc,"f0_method",e)},expression:"formData.payload.rvc.f0_method"}})],1)],1),r("el-form-item",{attrs:{label:" (index_rate)"}},[r("el-tooltip",{attrs:{content:"检索特征占比",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.rvc.index_rate,callback:function(e){t.$set(t.formData.payload.rvc,"index_rate",e)},expression:"formData.payload.rvc.index_rate"}})],1)],1),r("el-form-item",{attrs:{label:" (filter_radius)"}},[r("el-tooltip",{attrs:{content:"滤波半径",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.rvc.filter_radius,callback:function(e){t.$set(t.formData.payload.rvc,"filter_radius",e)},expression:"formData.payload.rvc.filter_radius"}})],1)],1),r("el-form-item",{attrs:{label:" (rms_mix_rate)"}},[r("el-tooltip",{attrs:{content:"输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.rvc.rms_mix_rate,callback:function(e){t.$set(t.formData.payload.rvc,"rms_mix_rate",e)},expression:"formData.payload.rvc.rms_mix_rate"}})],1)],1),r("el-form-item",{attrs:{label:"(resample_sr)"}},[r("el-tooltip",{attrs:{content:"后处理重采样至最终采样率，0为不进行重采样 ",placement:"top"}},[r("el-input",{model:{value:t.formData.payload.rvc.resample_sr,callback:function(e){t.$set(t.formData.payload.rvc,"resample_sr",e)},expression:"formData.payload.rvc.resample_sr"}})],1)],1)],1)])],1)],1)],1)],1)],1)],1)},o=[]},c973:function(t,e,r){function a(t,e,r,a,o,n,i){try{var l=t[n](i),c=l.value}catch(s){return void r(s)}l.done?e(c):Promise.resolve(c).then(a,o)}function o(t){return function(){var e=this,r=arguments;return new Promise((function(o,n){var i=t.apply(e,r);function l(t){a(i,o,n,l,c,"next",t)}function c(t){a(i,o,n,l,c,"throw",t)}l(void 0)}))}}r("d3b7"),t.exports=o,t.exports.__esModule=!0,t.exports["default"]=t.exports},e5a8:function(t,e,r){"use strict";var a=r("4ea4").default;Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0;var o=a(r("7037"));r("b64b"),r("fb6a"),r("d3b7");var n=a(r("bc3a"));n.default.defaults.headers["Content-Type"]="application/json;charset=utf-8";var i=n.default.create({baseURL:"/",timeout:3e4,withCredentials:!1});i.interceptors.request.use((function(t){if("get"===t.method&&t.params){for(var e=t.url+"?",r=0,a=Object.keys(t.params);r<a.length;r++){var n=a[r],i=t.params[n],l=encodeURIComponent(n)+"=";if(null!==i&&"undefined"!==typeof i)if("object"===(0,o.default)(i))for(var c=0,s=Object.keys(i);c<s.length;c++){var u=s[c],f=n+"["+u+"]",p=encodeURIComponent(f)+"=";e+=p+encodeURIComponent(i[u])+"&"}else e+=l+encodeURIComponent(i)+"&"}e=e.slice(0,-1),t.params={},t.url=e}return t}),(function(t){Promise.reject(t)}));var l=i;e.default=l}}]);