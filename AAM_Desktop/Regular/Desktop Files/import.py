import re
import requests
import os

# Example HTML (replace with your copied portion)
html_content = """
<html lang="en"><head><style class="darkreader darkreader--text" media="screen"></style><style class="darkreader darkreader--invert" media="screen"></style><style class="darkreader darkreader--inline" media="screen">[data-darkreader-inline-bgcolor] {
  background-color: var(--darkreader-inline-bgcolor) !important;
}
[data-darkreader-inline-bgimage] {
  background-image: var(--darkreader-inline-bgimage) !important;
}
[data-darkreader-inline-border] {
  border-color: var(--darkreader-inline-border) !important;
}
[data-darkreader-inline-border-bottom] {
  border-bottom-color: var(--darkreader-inline-border-bottom) !important;
}
[data-darkreader-inline-border-left] {
  border-left-color: var(--darkreader-inline-border-left) !important;
}
[data-darkreader-inline-border-right] {
  border-right-color: var(--darkreader-inline-border-right) !important;
}
[data-darkreader-inline-border-top] {
  border-top-color: var(--darkreader-inline-border-top) !important;
}
[data-darkreader-inline-boxshadow] {
  box-shadow: var(--darkreader-inline-boxshadow) !important;
}
[data-darkreader-inline-color] {
  color: var(--darkreader-inline-color) !important;
}
[data-darkreader-inline-fill] {
  fill: var(--darkreader-inline-fill) !important;
}
[data-darkreader-inline-stroke] {
  stroke: var(--darkreader-inline-stroke) !important;
}
[data-darkreader-inline-outline] {
  outline-color: var(--darkreader-inline-outline) !important;
}
[data-darkreader-inline-stopcolor] {
  stop-color: var(--darkreader-inline-stopcolor) !important;
}</style><style class="darkreader darkreader--variables" media="screen">:root {
   --darkreader-neutral-background: #202223;
   --darkreader-neutral-text: #e3dfda;
   --darkreader-selection-background: #0d59b5;
   --darkreader-selection-text: #f2f0ed;
}</style><style class="darkreader darkreader--root-vars" media="screen" data-single-file-stylesheet="4"></style><style type="text/css">.tippy-touch{cursor:pointer!important}.tippy-notransition{transition:none!important}.tippy-popper{max-width:350px;-webkit-perspective:700px;perspective:700px;z-index:9999;outline:0;transition-timing-function:cubic-bezier(.165,.84,.44,1);pointer-events:none;line-height:1.4}.tippy-popper[data-html]{max-width:96%;max-width:calc(100% - 20px)}.tippy-popper[x-placement^=top] .tippy-backdrop{border-radius:40% 40% 0 0}.tippy-popper[x-placement^=top] .tippy-roundarrow{bottom:-8px;-webkit-transform-origin:50% 0;transform-origin:50% 0}.tippy-popper[x-placement^=top] .tippy-roundarrow svg{position:absolute;left:0;-webkit-transform:rotate(180deg);transform:rotate(180deg)}.tippy-popper[x-placement^=top] .tippy-arrow{border-top:7px solid #333;border-right:7px solid transparent;border-left:7px solid transparent;bottom:-7px;margin:0 6px;-webkit-transform-origin:50% 0;transform-origin:50% 0}.tippy-popper[x-placement^=top] .tippy-backdrop{-webkit-transform-origin:0 90%;transform-origin:0 90%}.tippy-popper[x-placement^=top] .tippy-backdrop[data-state=visible]{-webkit-transform:scale(6) translate(-50%,25%);transform:scale(6) translate(-50%,25%);opacity:1}.tippy-popper[x-placement^=top] .tippy-backdrop[data-state=hidden]{-webkit-transform:scale(1) translate(-50%,25%);transform:scale(1) translate(-50%,25%);opacity:0}.tippy-popper[x-placement^=top] [data-animation=shift-toward][data-state=visible]{opacity:1;-webkit-transform:translateY(-10px);transform:translateY(-10px)}.tippy-popper[x-placement^=top] [data-animation=shift-toward][data-state=hidden]{opacity:0;-webkit-transform:translateY(-20px);transform:translateY(-20px)}.tippy-popper[x-placement^=top] [data-animation=perspective]{-webkit-transform-origin:bottom;transform-origin:bottom}.tippy-popper[x-placement^=top] [data-animation=perspective][data-state=visible]{opacity:1;-webkit-transform:translateY(-10px) rotateX(0);transform:translateY(-10px) rotateX(0)}.tippy-popper[x-placement^=top] [data-animation=perspective][data-state=hidden]{opacity:0;-webkit-transform:translateY(0) rotateX(90deg);transform:translateY(0) rotateX(90deg)}.tippy-popper[x-placement^=top] [data-animation=fade][data-state=visible]{opacity:1;-webkit-transform:translateY(-10px);transform:translateY(-10px)}.tippy-popper[x-placement^=top] [data-animation=fade][data-state=hidden]{opacity:0;-webkit-transform:translateY(-10px);transform:translateY(-10px)}.tippy-popper[x-placement^=top] [data-animation=shift-away][data-state=visible]{opacity:1;-webkit-transform:translateY(-10px);transform:translateY(-10px)}.tippy-popper[x-placement^=top] [data-animation=shift-away][data-state=hidden]{opacity:0;-webkit-transform:translateY(0);transform:translateY(0)}.tippy-popper[x-placement^=top] [data-animation=scale][data-state=visible]{opacity:1;-webkit-transform:translateY(-10px) scale(1);transform:translateY(-10px) scale(1)}.tippy-popper[x-placement^=top] [data-animation=scale][data-state=hidden]{opacity:0;-webkit-transform:translateY(0) scale(0);transform:translateY(0) scale(0)}.tippy-popper[x-placement^=bottom] .tippy-backdrop{border-radius:0 0 30% 30%}.tippy-popper[x-placement^=bottom] .tippy-roundarrow{top:-8px;-webkit-transform-origin:50% 100%;transform-origin:50% 100%}.tippy-popper[x-placement^=bottom] .tippy-roundarrow svg{position:absolute;left:0;-webkit-transform:rotate(0);transform:rotate(0)}.tippy-popper[x-placement^=bottom] .tippy-arrow{border-bottom:7px solid #333;border-right:7px solid transparent;border-left:7px solid transparent;top:-7px;margin:0 6px;-webkit-transform-origin:50% 100%;transform-origin:50% 100%}.tippy-popper[x-placement^=bottom] .tippy-backdrop{-webkit-transform-origin:0 -90%;transform-origin:0 -90%}.tippy-popper[x-placement^=bottom] .tippy-backdrop[data-state=visible]{-webkit-transform:scale(6) translate(-50%,-125%);transform:scale(6) translate(-50%,-125%);opacity:1}.tippy-popper[x-placement^=bottom] .tippy-backdrop[data-state=hidden]{-webkit-transform:scale(1) translate(-50%,-125%);transform:scale(1) translate(-50%,-125%);opacity:0}.tippy-popper[x-placement^=bottom] [data-animation=shift-toward][data-state=visible]{opacity:1;-webkit-transform:translateY(10px);transform:translateY(10px)}.tippy-popper[x-placement^=bottom] [data-animation=shift-toward][data-state=hidden]{opacity:0;-webkit-transform:translateY(20px);transform:translateY(20px)}.tippy-popper[x-placement^=bottom] [data-animation=perspective]{-webkit-transform-origin:top;transform-origin:top}.tippy-popper[x-placement^=bottom] [data-animation=perspective][data-state=visible]{opacity:1;-webkit-transform:translateY(10px) rotateX(0);transform:translateY(10px) rotateX(0)}.tippy-popper[x-placement^=bottom] [data-animation=perspective][data-state=hidden]{opacity:0;-webkit-transform:translateY(0) rotateX(-90deg);transform:translateY(0) rotateX(-90deg)}.tippy-popper[x-placement^=bottom] [data-animation=fade][data-state=visible]{opacity:1;-webkit-transform:translateY(10px);transform:translateY(10px)}.tippy-popper[x-placement^=bottom] [data-animation=fade][data-state=hidden]{opacity:0;-webkit-transform:translateY(10px);transform:translateY(10px)}.tippy-popper[x-placement^=bottom] [data-animation=shift-away][data-state=visible]{opacity:1;-webkit-transform:translateY(10px);transform:translateY(10px)}.tippy-popper[x-placement^=bottom] [data-animation=shift-away][data-state=hidden]{opacity:0;-webkit-transform:translateY(0);transform:translateY(0)}.tippy-popper[x-placement^=bottom] [data-animation=scale][data-state=visible]{opacity:1;-webkit-transform:translateY(10px) scale(1);transform:translateY(10px) scale(1)}.tippy-popper[x-placement^=bottom] [data-animation=scale][data-state=hidden]{opacity:0;-webkit-transform:translateY(0) scale(0);transform:translateY(0) scale(0)}.tippy-popper[x-placement^=left] .tippy-backdrop{border-radius:50% 0 0 50%}.tippy-popper[x-placement^=left] .tippy-roundarrow{right:-16px;-webkit-transform-origin:33.33333333% 50%;transform-origin:33.33333333% 50%}.tippy-popper[x-placement^=left] .tippy-roundarrow svg{position:absolute;left:0;-webkit-transform:rotate(90deg);transform:rotate(90deg)}.tippy-popper[x-placement^=left] .tippy-arrow{border-left:7px solid #333;border-top:7px solid transparent;border-bottom:7px solid transparent;right:-7px;margin:3px 0;-webkit-transform-origin:0 50%;transform-origin:0 50%}.tippy-popper[x-placement^=left] .tippy-backdrop{-webkit-transform-origin:100% 0;transform-origin:100% 0}.tippy-popper[x-placement^=left] .tippy-backdrop[data-state=visible]{-webkit-transform:scale(6) translate(40%,-50%);transform:scale(6) translate(40%,-50%);opacity:1}.tippy-popper[x-placement^=left] .tippy-backdrop[data-state=hidden]{-webkit-transform:scale(1.5) translate(40%,-50%);transform:scale(1.5) translate(40%,-50%);opacity:0}.tippy-popper[x-placement^=left] [data-animation=shift-toward][data-state=visible]{opacity:1;-webkit-transform:translateX(-10px);transform:translateX(-10px)}.tippy-popper[x-placement^=left] [data-animation=shift-toward][data-state=hidden]{opacity:0;-webkit-transform:translateX(-20px);transform:translateX(-20px)}.tippy-popper[x-placement^=left] [data-animation=perspective]{-webkit-transform-origin:right;transform-origin:right}.tippy-popper[x-placement^=left] [data-animation=perspective][data-state=visible]{opacity:1;-webkit-transform:translateX(-10px) rotateY(0);transform:translateX(-10px) rotateY(0)}.tippy-popper[x-placement^=left] [data-animation=perspective][data-state=hidden]{opacity:0;-webkit-transform:translateX(0) rotateY(-90deg);transform:translateX(0) rotateY(-90deg)}.tippy-popper[x-placement^=left] [data-animation=fade][data-state=visible]{opacity:1;-webkit-transform:translateX(-10px);transform:translateX(-10px)}.tippy-popper[x-placement^=left] [data-animation=fade][data-state=hidden]{opacity:0;-webkit-transform:translateX(-10px);transform:translateX(-10px)}.tippy-popper[x-placement^=left] [data-animation=shift-away][data-state=visible]{opacity:1;-webkit-transform:translateX(-10px);transform:translateX(-10px)}.tippy-popper[x-placement^=left] [data-animation=shift-away][data-state=hidden]{opacity:0;-webkit-transform:translateX(0);transform:translateX(0)}.tippy-popper[x-placement^=left] [data-animation=scale][data-state=visible]{opacity:1;-webkit-transform:translateX(-10px) scale(1);transform:translateX(-10px) scale(1)}.tippy-popper[x-placement^=left] [data-animation=scale][data-state=hidden]{opacity:0;-webkit-transform:translateX(0) scale(0);transform:translateX(0) scale(0)}.tippy-popper[x-placement^=right] .tippy-backdrop{border-radius:0 50% 50% 0}.tippy-popper[x-placement^=right] .tippy-roundarrow{left:-16px;-webkit-transform-origin:66.66666666% 50%;transform-origin:66.66666666% 50%}.tippy-popper[x-placement^=right] .tippy-roundarrow svg{position:absolute;left:0;-webkit-transform:rotate(-90deg);transform:rotate(-90deg)}.tippy-popper[x-placement^=right] .tippy-arrow{border-right:7px solid #333;border-top:7px solid transparent;border-bottom:7px solid transparent;left:-7px;margin:3px 0;-webkit-transform-origin:100% 50%;transform-origin:100% 50%}.tippy-popper[x-placement^=right] .tippy-backdrop{-webkit-transform-origin:-100% 0;transform-origin:-100% 0}.tippy-popper[x-placement^=right] .tippy-backdrop[data-state=visible]{-webkit-transform:scale(6) translate(-140%,-50%);transform:scale(6) translate(-140%,-50%);opacity:1}.tippy-popper[x-placement^=right] .tippy-backdrop[data-state=hidden]{-webkit-transform:scale(1.5) translate(-140%,-50%);transform:scale(1.5) translate(-140%,-50%);opacity:0}.tippy-popper[x-placement^=right] [data-animation=shift-toward][data-state=visible]{opacity:1;-webkit-transform:translateX(10px);transform:translateX(10px)}.tippy-popper[x-placement^=right] [data-animation=shift-toward][data-state=hidden]{opacity:0;-webkit-transform:translateX(20px);transform:translateX(20px)}.tippy-popper[x-placement^=right] [data-animation=perspective]{-webkit-transform-origin:left;transform-origin:left}.tippy-popper[x-placement^=right] [data-animation=perspective][data-state=visible]{opacity:1;-webkit-transform:translateX(10px) rotateY(0);transform:translateX(10px) rotateY(0)}.tippy-popper[x-placement^=right] [data-animation=perspective][data-state=hidden]{opacity:0;-webkit-transform:translateX(0) rotateY(90deg);transform:translateX(0) rotateY(90deg)}.tippy-popper[x-placement^=right] [data-animation=fade][data-state=visible]{opacity:1;-webkit-transform:translateX(10px);transform:translateX(10px)}.tippy-popper[x-placement^=right] [data-animation=fade][data-state=hidden]{opacity:0;-webkit-transform:translateX(10px);transform:translateX(10px)}.tippy-popper[x-placement^=right] [data-animation=shift-away][data-state=visible]{opacity:1;-webkit-transform:translateX(10px);transform:translateX(10px)}.tippy-popper[x-placement^=right] [data-animation=shift-away][data-state=hidden]{opacity:0;-webkit-transform:translateX(0);transform:translateX(0)}.tippy-popper[x-placement^=right] [data-animation=scale][data-state=visible]{opacity:1;-webkit-transform:translateX(10px) scale(1);transform:translateX(10px) scale(1)}.tippy-popper[x-placement^=right] [data-animation=scale][data-state=hidden]{opacity:0;-webkit-transform:translateX(0) scale(0);transform:translateX(0) scale(0)}.tippy-tooltip{position:relative;color:#fff;border-radius:4px;font-size:.9rem;padding:.3rem .6rem;text-align:center;will-change:transform;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;background-color:#333}.tippy-tooltip[data-size=small]{padding:.2rem .4rem;font-size:.75rem}.tippy-tooltip[data-size=large]{padding:.4rem .8rem;font-size:1rem}.tippy-tooltip[data-animatefill]{overflow:hidden;background-color:transparent}.tippy-tooltip[data-animatefill] .tippy-content{transition:-webkit-clip-path cubic-bezier(.46,.1,.52,.98);transition:clip-path cubic-bezier(.46,.1,.52,.98);transition:clip-path cubic-bezier(.46,.1,.52,.98),-webkit-clip-path cubic-bezier(.46,.1,.52,.98)}.tippy-tooltip[data-interactive],.tippy-tooltip[data-interactive] path{pointer-events:auto}.tippy-tooltip[data-inertia][data-state=visible]{transition-timing-function:cubic-bezier(.53,2,.36,.85)}.tippy-tooltip[data-inertia][data-state=hidden]{transition-timing-function:ease}.tippy-arrow,.tippy-roundarrow{position:absolute;width:0;height:0}.tippy-roundarrow{width:24px;height:8px;fill:#333;pointer-events:none}.tippy-backdrop{position:absolute;will-change:transform;background-color:#333;border-radius:50%;width:26%;left:50%;top:50%;z-index:-1;transition:all cubic-bezier(.46,.1,.52,.98);-webkit-backface-visibility:hidden;backface-visibility:hidden}.tippy-backdrop:after{content:"";float:left;padding-top:100%}body:not(.tippy-touch) .tippy-tooltip[data-animatefill][data-state=visible] .tippy-content{-webkit-clip-path:ellipse(100% 100% at 50% 50%);clip-path:ellipse(100% 100% at 50% 50%)}body:not(.tippy-touch) .tippy-tooltip[data-animatefill][data-state=hidden] .tippy-content{-webkit-clip-path:ellipse(5% 50% at 50% 50%);clip-path:ellipse(5% 50% at 50% 50%)}body:not(.tippy-touch) .tippy-popper[x-placement=right] .tippy-tooltip[data-animatefill][data-state=visible] .tippy-content{-webkit-clip-path:ellipse(135% 100% at 0 50%);clip-path:ellipse(135% 100% at 0 50%)}body:not(.tippy-touch) .tippy-popper[x-placement=right] .tippy-tooltip[data-animatefill][data-state=hidden] .tippy-content{-webkit-clip-path:ellipse(40% 100% at 0 50%);clip-path:ellipse(40% 100% at 0 50%)}body:not(.tippy-touch) .tippy-popper[x-placement=left] .tippy-tooltip[data-animatefill][data-state=visible] .tippy-content{-webkit-clip-path:ellipse(135% 100% at 100% 50%);clip-path:ellipse(135% 100% at 100% 50%)}body:not(.tippy-touch) .tippy-popper[x-placement=left] .tippy-tooltip[data-animatefill][data-state=hidden] .tippy-content{-webkit-clip-path:ellipse(40% 100% at 100% 50%);clip-path:ellipse(40% 100% at 100% 50%)}@media (max-width:360px){.tippy-popper{max-width:96%;max-width:calc(100% - 20px)}}</style>
    <meta charset="utf-8">
<title>TOPICAL PAST PAPER QUESTIONS | exam-mate</title>
<meta name="title" content="TOPICAL PAST PAPER QUESTIONS | exam-mate">

<meta property="og:title" content="Exam-Mate: Prepares Students for Exam, Asists Teachers to Make Exam">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta content="exam-mate is an exam preparation tool containing a bank of IGCSE, A-Level, IB, AQA and OCR Topical questions and yearly past papers exams. With exam-mate you are able to build online exams easily using our question bank database." name="description">
<meta name="keywords" content="Topical Past Papers,Topical Past Papers Books,IGCSE Past Papers Yearly,IGCSE Add Math Topical Past Papers,IGCSE Chemistry Topical Past Papers,IGCSE English Topical Past Papers,IGCSE Physics Topical Past Papers,IGCSE Math Topical Past Papers,IGCSE Topical Past Papers,IGCSE ICT Topical Past Papers,IGCSE Economics Topical Past Papers,A-Level Topical Past Papers,IGCSE Biology Topical Past Papers,IGCSE Physics Past Papers,A-Level Past Papers Yearly,A-Level Add Pure Math Topical Past Papers,IA-Level Chemistry Topical Past Papers,A-Level Physics Topical Past Papers,IA-Level Math Topical Past Papers,A-Level Topical Past Papers Books,A-Level Biology Topical Past Papers,A-Level Computer Science Topical Past Papers,A-Level Physics Past Papers,Build exam,Online exam builder,exam builder,Edexcel Past Papers,Edexcel Topical Past Papers Books,Edexcel Topical Past Papers ,Cambridge Topical Past Papers Books,Edexcel IAL,Cambridge Topical IGCSE Past Papers,Mark schemes Past Papers,Edexcel International GCSE,IB Topical Past Papers,IB Topical Past Papers Books,IB Past Papers Yearly,IB Math HL Topical Past Papers,IB Chemistry HL Topical Past Papers,IB Physics HL Topical Past Papers,IB Math HL Topical Past Papers ,IB Math SL Topical Past Papers,IB Chemistry SL Topical Past Papers,IB Physics SL Topical Past Papers,IB Math SL Topical Past Papers ,IB Question Bank,IGCSE Question Bank,IA-Level Question Bank,IB Questions by topics,A-Level Questions by topics,IGCSE Questions by topics,IB Physics Questions by topics,IB Chemistry Questions by topics,IB Biology Questions by topics,IB Math Questions by topics,IGCSE Physics Questions by topics,IGCSE Chemistry Questions by topics,IGCSE Biology Questions by topics,IGCSE Math Questions by topics,A-Level Physics Questions by topics,A-Level Chemistry Questions by topics,A-Level Biology Questions by topics,A-Level Math Questions by topics,IGCSE Checkpoint Past Papers">
<meta content="exam-mate" name="author">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<!-- App favicon -->
<link rel="shortcut icon" href="https://www.exam-mate.com/storage/fav_icon.png">

<meta property="og:locale" content="en_US">
<meta property="og:type" content="website">
<meta property="og:title" content="Exam-Mate: Digital Learning Platform">
<meta property="og:description" content="exam-mate is an exam preparation and exam builder tool, containing a bank of topical and yearly past papers. It covers Cambridge IGCSE Past Papers, Edexcel International GCSE, Cambridge and Edexcel A Level and IAL along with their mark schemes. Students can use it to access questions related to topics, while teachers can use the software during teaching and to make exam papers easily.">
<meta property="og:url" content="http://exam-mate.com/">
<meta property="og:site_name" content="Exam-Mate">
<meta property="fb:app_id" content="Exam-mate-1690688471242502">
<meta property="og:image" content="https://www.exam-mate.com/storage/fav_icon.png">
<meta name="author" content="exam-mate.com">
<meta name="og:email" content="office@exam-mate.com">




    <!-- icons -->
<link href="https://www.exam-mate.com/assets/css/icons.min.css" rel="stylesheet" type="text/css">
<link rel="stylesheet" href="https://cdn-uicons.flaticon.com/uicons-regular-rounded/css/uicons-regular-rounded.css">
<link rel="stylesheet" href="https://cdn-uicons.flaticon.com/uicons-bold-rounded/css/uicons-bold-rounded.css">
<link rel="stylesheet" href="https://cdn-uicons.flaticon.com/uicons-solid-rounded/css/uicons-solid-rounded.css">
<link rel="stylesheet" as="style" onload="this.onload=null;this.rel='stylesheet'" href="https://www.exam-mate.com/assets/libs/jquery-toast-plugin/jquery-toast-plugin.min.css" type="text/css">
<noscript>
    <link rel="stylesheet" href="https://www.exam-mate.com/assets/libs/jquery-toast-plugin/jquery-toast-plugin.min.css"
        type="text/css" />
</noscript>



    <!-- App css -->
                                                                                    <link href="https://www.exam-mate.com/assets/css/default/bootstrap.min.css" rel="stylesheet" type="text/css" id="bs-default-stylesheet">
                        <link href="https://www.exam-mate.com/assets/css/default/app.min.css " rel="stylesheet" type="text/css" id="app-default-stylesheet">
                        <link href="https://www.exam-mate.com/assets/css/default/bootstrap-dark.min.css " rel="stylesheet" type="text/css" id="bs-dark-stylesheet" disabled="disabled">
                        <link href="https://www.exam-mate.com/assets/css/default/app-dark.min.css " rel="stylesheet" type="text/css" id="app-dark-stylesheet" disabled="disabled">
                                                            
    <style>[wire\:loading], [wire\:loading\.delay], [wire\:loading\.inline-block], [wire\:loading\.inline], [wire\:loading\.block], [wire\:loading\.flex], [wire\:loading\.table], [wire\:loading\.grid], [wire\:loading\.inline-flex] {display: none;}[wire\:loading\.delay\.shortest], [wire\:loading\.delay\.shorter], [wire\:loading\.delay\.short], [wire\:loading\.delay\.long], [wire\:loading\.delay\.longer], [wire\:loading\.delay\.longest] {display:none;}[wire\:offline] {display: none;}[wire\:dirty]:not(textarea):not(input):not(select) {display: none;}input:-webkit-autofill, select:-webkit-autofill, textarea:-webkit-autofill {animation-duration: 50000s;animation-name: livewireautofill;}@keyframes livewireautofill { from {} }</style>

<link type="text/css" rel="stylesheet" id="dark-mode-custom-link"><link type="text/css" rel="stylesheet" id="dark-mode-general-link"><style lang="en" type="text/css" id="dark-mode-custom-style"></style><style lang="en" type="text/css" id="dark-mode-native-style"></style><style lang="en" type="text/css" id="dark-mode-native-sheet"></style><style class="darkreader darkreader--override" media="screen"></style></head>


<body @keydown.window.prevent.meta.up="$('#prev-btn').click()" @keydown.window.prevent.meta.down="$('#next-btn').click()" x-data="" @keydown.window.prevent.enter="$('.filter-button').click()" class="menuitem-active" data-layout="{&quot;mode&quot;: &quot;light&quot;, &quot;width&quot;: &quot;fluid&quot;, &quot;menuPosition&quot;: &quot;fixed&quot;, &quot;sidebar&quot;: { &quot;color&quot;: &quot;light&quot;, &quot;size&quot;: &quot;default&quot;, &quot;showuser&quot;: false}, &quot;topbar&quot;: {&quot;color&quot;: &quot;dark&quot;}, &quot;showRightSidebarOnPageLoad&quot;: true}" data-sidebar-size="default" data-layout-width="fluid" data-layout-menu-position="fixed" data-sidebar-color="light" data-sidebar-showuser="false" data-topbar-color="dark" style="visibility: visible; opacity: 1;" data-new-gr-c-s-check-loaded="14.1235.0" data-gr-ext-installed="">
    <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-T23FMLK" height="0" width="0"
            style="display:none;visibility:hidden"></iframe></noscript>

    <script defer="" src="https://api.pirsch.io/pirsch.js" id="pirschjs" data-code="EPYuMRBnbAPlKLgANJBKg8CRv9NVDX78"></script>

    <!-- Begin page -->
    <div id="wrapper" class="show">
        <!-- Topbar Start -->
<div class="navbar-custom">
    <div class="container-fluid">
        <ul class="list-unstyled topnav-menu float-end mb-0">
            
            <li class="dropdown d-none d-lg-inline-block">
                <a class=" nav-link dropdown-toggle arrow-none waves-effect waves-light" href="https://www.exam-mate.com/affiliate" data-plugin="tippy" data-tippy-size="small" data-tippy-placement="bottom" data-tippy="" data-original-title="Affiliate Program">
                    <i class="fas fa-network-wired "></i>
                </a>
            </li>

            <li class="dropdown d-none d-lg-inline-block">
                <a class=" nav-link dropdown-toggle arrow-none waves-effect waves-light" href="https://www.exam-mate.com/faq" data-plugin="tippy" data-tippy-size="small" data-tippy-placement="bottom" data-tippy="" data-original-title="Help Center">
                    <i class="fas fa-question "></i>
                </a>
            </li>

            <li class="dropdown d-none d-lg-inline-block">
                <a class=" nav-link dropdown-toggle arrow-none waves-effect waves-light" href="https://www.exam-mate.com/tickets/create" data-plugin="tippy" data-tippy-size="small" data-tippy-placement="bottom" data-tippy="" data-original-title="Contact us">
                    <i class="fas fa-envelope"></i>
                </a>
            </li>

                                                <li class="dropdown notification-list topbar-dropdown" tabindex="0" data-plugin="tippy" data-tippy-size="small" data-tippy-placement="bottom" data-tippy="" data-original-title="Shopping Cart">
                <a wire:id="ZUhsIIeRMDPp0NhOrHkF" class=" nav-link dropdown-toggle waves-effect waves-light" href="https://www.exam-mate.com/cart">
    <i class="fe-shopping-cart noti-icon"></i>
    <span class="badge bg-danger rounded-circle noti-icon-badge">0</span>
</a>

<!-- Livewire Component wire-end:ZUhsIIeRMDPp0NhOrHkF -->            </li>

                        <li class="dropdown notification-list topbar-dropdown">
                <a class="nav-link dropdown-toggle nav-user me-0 waves-effect waves-light" data-bs-toggle="dropdown" href="#" role="button" aria-haspopup="false" aria-expanded="false">
                    <img src="https://www.exam-mate.com/storage/user_icon_default.svg" alt="user-image" class="rounded-circle">
                    <span class="pro-user-name ms-1">
                        
                        <i class="mdi mdi-chevron-down"></i>
                    </span>
                </a>
                <div class="dropdown-menu dropdown-menu-end profile-dropdown ">
                    <!-- item-->
                    <div class="dropdown-header noti-title">
                        <h6 class="text-overflow m-0">Welcome !</h6>
                    </div>

                    <!-- item-->
                    <a href="https://www.exam-mate.com/user/profile" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-user text-danger"></i>
                        <span>My Account</span>
                    </a>
                                                                                                    <div class="dropdown-divider"></div>
                                                                                                                                            <a href="https://www.exam-mate.com/myexams" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-folder text-danger"></i>
                        <span>My Built Exams</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/list" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-list text-danger"></i>
                        <span>My Question Lists</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/mypdfs" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-book-open text-danger"></i>
                        <span>My Generated PDFs</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/myfavoritequestions" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-heart text-danger"></i>
                        <span>My Favorite Questions</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/myfavoritepastpapers" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-heart text-danger"></i>
                        <span>My Favorite Past Papers</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/orders" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-activity text-danger"></i>
                        <span>My Orders</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/orders/paid" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-calendar text-danger"></i>
                        <span>My Purchased History</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/myproducts" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-briefcase text-danger"></i>
                        <span>My Products</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/mymockexams" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-edit text-danger"></i>
                        <span>My Mock Exams</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/tickets" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-mail text-danger"></i>
                        <span>My Tickets</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/affiliate/dashboard" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-share-2 text-danger"></i>
                        <span>My Affilliate Dashboard</span>
                    </a>
                                                                                                                        <a href="https://www.exam-mate.com/subscriptions" class="dropdown-item notify-item right-bar-item">
                        <i class="fe-user-check text-danger"></i>
                        <span>My Subscriptions Status</span>
                    </a>
                                                                                                    <div class="dropdown-divider"></div>
                                                                                                    <!-- item-->
                    <form method="POST" action="https://www.exam-mate.com/logout">
                        <input type="hidden" name="_token" value="Hl3szbwt5vFLH8eskzLZalI8chsWb3Iw7opK5Ieq">                        <a href="javascript:void(0);" class="dropdown-item notify-item right-bar-item" onclick="event.preventDefault();
                                                this.closest('form').submit();">
                            <i class="fe-log-out text-danger"></i>
                            <span>Logout</span>
                        </a>
                    </form>

                </div>
            </li>
                                </ul>

        <!-- LOGO -->
        <div class="logo-box">
            <a href="https://www.exam-mate.com" class="logo logo-dark text-center">
                <span class="logo-sm">
                    <img src="https://www.exam-mate.com/storage/logo.png" alt="" height="50">
                </span>
                <span class="logo-lg">
                    <img src="https://www.exam-mate.com/storage/logo_small.png" alt="" height="40">
                </span>
            </a>

            <a href="https://www.exam-mate.com" class="logo logo-light text-center">
                <span class="logo-sm">
                    <img src="https://www.exam-mate.com/storage/logo_small.png" alt="" height="40">
                </span>
                <span class="logo-lg">
                    <img src="https://www.exam-mate.com/storage/logo.png" alt="" height="50">
                </span>
            </a>
        </div>

        <ul class="list-unstyled topnav-menu topnav-menu-left m-0">
            <li>
                <button class="button-menu-mobile waves-effect waves-light">
                    <i class="fe-menu"></i>
                </button>
            </li>

            <li>
                <!-- Mobile menu toggle (Horizontal Layout)-->
                <a class="navbar-toggle nav-link" data-bs-toggle="collapse" data-bs-target="#topnav-menu-content">
                    <div class="lines">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </a>
                <!-- End mobile menu toggle-->
            </li>
                                                            <li class="d-none d-md-table-cell">
                <a class=" nav-link waves-effect waves-light" href="https://www.exam-mate.com/topicalpastpapers_books">
                    <span class="p-1 top-bar-menu">
                        <span>Topical Past Papers eBooks (SHOP)</span>
                    </span>
                </a>

            </li>
                                                            <li class="dropdown  d-md-block">
                <a class="nav-link dropdown-toggle waves-effect waves-light" data-bs-toggle="dropdown" href="#" role="button" aria-haspopup="false" aria-expanded="false">
                    <span>Subscriptions</span>
                    <i class="mdi mdi-chevron-down"></i>
                </a>
                <div class="dropdown-menu">
                    <!-- item-->
                                                            <a href="https://www.exam-mate.com/topicalpastpapers/pricing" class="dropdown-item">
                        <i class="fe-user-check me-1 text-danger"></i>
                        <span>Access to Topical Past Papers</span>
                    </a>
                                                                                <a href="https://www.exam-mate.com/mcq/pricing" class="dropdown-item">
                        <i class="fe-user-check me-1 text-danger"></i>
                        <span>Online MCQ Past Papers</span>
                    </a>
                                                                                <a href="https://www.exam-mate.com/exams/pricing" class="dropdown-item">
                        <i class="fe-user-check me-1 text-danger"></i>
                        <span>Build Exam OR Worksheet</span>
                    </a>
                                                                                <a href="https://www.exam-mate.com/extra_pdf/pricing" class="dropdown-item">
                        <i class="fe-user-check me-1 text-danger"></i>
                        <span>Extra PDF</span>
                    </a>
                                                                                <a href="https://www.exam-mate.com/remove_branding/pricing" class="dropdown-item">
                        <i class="fe-user-check me-1 text-danger"></i>
                        <span>Remove Branding</span>
                    </a>
                                                                                <a href="https://www.exam-mate.com/mockexams/pricing" class="dropdown-item">
                        <i class="fe-user-check me-1 text-danger"></i>
                        <span>Mock Exam</span>
                    </a>
                                                                                <a href="https://www.exam-mate.com/school/pricing" class="dropdown-item">
                        <i class="fe-user-check me-1 text-danger"></i>
                        <span>School Pricing</span>
                    </a>
                                                        </div>
            </li>
                                            </ul>
        <div class="clearfix"></div>
    </div>
</div>
<!-- end Topbar -->

        <div wire:id="KdmpDlNqrxgD7o5ZCw9K">
    
    <div id="new-exam-modal" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="CreateNewExam" aria-hidden="true" style="display: none;" wire:ignore.self="create-exam-button">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-body p-4 pt-0 pb-2">
                    <div class="text-start mb-3">
                        <span class="fi fi-br-bulb-nib fa-2x mb-2 d-inline-block text-primary"> </span>
                        <h4 class="modal-title fs-3 fw-light">CREATE NEW EXAM</h4>
                    </div>
                    <div class="tab-pane" id="create-exam-tab" wire:ignore.self="add-exam-button">
                        <div>
                                                    </div>
                        <div class="mb-3">
                            <label for="exam-name" class="form-label">Exam Name</label>
                            <input wire:model="examName" type="text" placeholder="My exam" id="exam-name" class="form-control" required="">
                                                        <div class="text-start mt-2">
                                                                    <button class="mt-3 btn btn-primary waves-effect waves-light" wire:key="create-exam-button" wire:click="createExam">
                                        <span class="btn-label"><i class="fas fa-plus"></i></span>
                                        Create Exam
                                    </button>
                                                                                                <button class="mt-3 btn btn-danger waves-effect waves-light" data-bs-dismiss="modal" aria-label="Close">
                                    <span class="btn-label"><i class="fas fa-times"></i></span>
                                    Cancel
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div><!-- /.modal -->
    
</div>

<!-- Livewire Component wire-end:KdmpDlNqrxgD7o5ZCw9K --><div wire:id="cNdhEqwhOQMiPlvv0qMG">
    
    <div id="new-list-modal" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="CreateNewList" aria-hidden="true" style="display: none;" wire:ignore.self="create-list-button">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-body p-4 pt-0 pb-2">
                    <div class="text-start mb-3">
                        <span class="fi fi-br-bulb-nib fa-2x mb-2 d-inline-block text-primary"> </span>
                        <h4 class="modal-title fs-3 fw-light">Create New Question List</h4>
                    </div>
                    <div class="tab-pane" id="create-list-tab" wire:ignore.self="add-list-button">
                        <div>
                                                    </div>
                        <div class="mb-3">
                            <label for="list-name" class="form-label">Question List Name</label>
                            <input wire:model="listName" type="text" placeholder="My List" id="list-name" class="form-control" required="">
                                                        <div class="text-start mt-2">
                                                                    <button class="mt-3 btn btn-primary waves-effect waves-light" wire:key="create-list-button" wire:click="createList">
                                        <span class="btn-label"><i class="fas fa-plus"></i></span> Create Question list
                                    </button>
                                                                                                <button class="mt-3 btn btn-danger waves-effect waves-light" data-bs-dismiss="modal" aria-label="Close">
                                    <span class="btn-label"><i class="fas fa-times"></i></span>
                                    Cancel
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div><!-- /.modal -->
    
</div>

<!-- Livewire Component wire-end:cNdhEqwhOQMiPlvv0qMG -->
<div class="left-side-menu">

    <div class="h-100 menuitem-active" data-simplebar="init"><div class="simplebar-wrapper" style="margin: 0px;"><div class="simplebar-height-auto-observer-wrapper"><div class="simplebar-height-auto-observer"></div></div><div class="simplebar-mask"><div class="simplebar-offset" style="right: 0px; bottom: 0px;"><div class="simplebar-content-wrapper" style="height: 100%; overflow: hidden scroll;"><div class="simplebar-content" style="padding: 0px;">
                    <!-- User box -->
            <div class="text-center user-box">
                <img src="https://www.exam-mate.com/storage/user_icon_default.svg" alt="user-img" title="Mat Helme" class="rounded-circle avatar-md">
                <div class="dropdown">
                    <a href="javascript: void(0);" class="mt-2 mb-1 text-dark dropdown-toggle h5 d-block" data-bs-toggle="dropdown">Shatila</a>
                    <div class="dropdown-menu user-pro-dropdown">

                        <!-- item-->
                        <a href="https://www.exam-mate.com/user/profile" class="dropdown-item notify-item">
                            <i class="fe-user me-1"></i>
                            <span>My Account</span>
                        </a>

                        <!-- item-->
                        <a href="javascript:void(0);" class="dropdown-item notify-item">
                            <i class="fe-log-out me-1"></i>
                            <span>Logout</span>
                        </a>

                    </div>
                </div>
                <p class="text-muted">Student</p>
            </div>
        
        <!--- Sidemenu -->
        <div id="sidebar-menu" class="show">

            <ul id="side-menu">
                                                                                                                <li>
                                <a href="https://www.exam-mate.com/pastpapers" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-file"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> PAST PAPERS
                                                                                            </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        free, Pdf, downloadable</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li class="menuitem-active">
                                <a href="https://www.exam-mate.com/topicalpastpapers" class="side-menu-item-container active">
                                    <i class="align-middle  fi fi-rr-search-alt"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> TOPICAL PAST PAPERS
                                                                                            </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        not free, online, not downloadable</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li>
                                <a href="https://www.exam-mate.com/mcq" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-file-chart-pie"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> MCQ PAST PAPERS
                                                                                            </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        not free, online practice</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li>
                                <a href="https://www.exam-mate.com/untopicalpastpapers" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-document"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> NON-TOPICAL PAST PAPERS
                                                                                            </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        free, online, not downloadable</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li>
                                <a href="https://www.exam-mate.com/solved_pastpapers" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-edit"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> SOLVED PAST PAPERS
                                                                                            </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        not free, Pdf, downloadable</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li>
                                <a href="https://www.exam-mate.com/questionbank" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-database"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> EXAM-MATE QUESTION BANK
                                                                                            </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        free, online, not downloadable</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li>
                                <a href="https://www.exam-mate.com/testprep" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-test"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> ACT, SAT, AP
                                                                                            </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        free, online, not downloadable</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li>
                                <a href="#" class="side-menu-item-container" data-bs-toggle="modal" data-bs-target="#new-exam-modal">
                                    <i class="align-middle  fi fi-rr-apps-add"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> BUILD EXAM OR WORKSHEET </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        not free, convert to pdf, printable</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li>
                                <a href="#" class="side-menu-item-container" data-bs-toggle="modal" data-bs-target="#new-list-modal">
                                    <i class="align-middle  fi fi-rr-add"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> BUILD QUESTION LIST </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        free, online, not downloadable</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li>
                                <a href="https://topicalpastpapers.com/free-topical-worksheets/?" class="side-menu-item-container" target="_blank">
                                    <i class="align-middle  fi fi-rr-book"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> FREE TOPICAL BOOKS </span>
                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                    free topical worksheets</em> </h6>
                                        </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li>
                                <a href="https://www.exam-mate.com/mockexams" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-wave-sine"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> MOCK EXAMS
                                                                                            </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        not free, online</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li>
                                <a href="https://www.exam-mate.com/golden_notes" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-comment-pen"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> GOLDEN NOTES
                                                                                            </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        free, online, not downloadable</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                                                                            <li class="d-md-none">
                                <a href="https://www.exam-mate.com/topicalpastpapers_books" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-books"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> TOPICAL PAST PAPER eBOOKS </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        topical e-books</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li class="d-md-none">
                                <a href="https://www.exam-mate.com/school" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-school"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> SCHOOL DASHBOARD </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        school dashboard</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                                        <li class="d-md-none">
                                <a href="https://www.exam-mate.com/faq" class="side-menu-item-container">
                                    <i class="align-middle  fi fi-rr-question-square"></i>
                                    <span>
                                        <div class="d-inline-block">
                                            <span class="d-block side-menu-item-title"> HELP CENTER </span>
                                                                                            <h6 class="m-0 side-menu-item-desc"> <em>
                                                        Frequently Asked Questions</em> </h6>
                                                                                    </div>
                                    </span>
                                </a>
                            </li>
                                                                                                <li>
                        <a href="https://exam-mate.com/for-schools" class="side-menu-item-container">
                            <img class="mw-100" src="https://www.exam-mate.com/storage/banner_left.webp">
                        </a>
                    </li>
                            </ul>


        </div>
        <!-- End Sidebar -->

        <div class="clearfix"></div>

    </div></div></div></div><div class="simplebar-placeholder" style="width: auto; height: 606px;"></div></div><div class="simplebar-track simplebar-horizontal" style="visibility: hidden;"><div class="simplebar-scrollbar" style="width: 0px; display: none;"></div></div><div class="simplebar-track simplebar-vertical" style="visibility: visible;"><div class="simplebar-scrollbar" style="height: 134px; transform: translate3d(0px, 0px, 0px); display: block;"></div></div></div>
    <!-- Sidebar -left -->

</div>
<!-- Left Sidebar End -->

        <!-- ============================================================== -->
        <!-- Start Page Content here -->
        <!-- ============================================================== -->

        <div class="content-page">
                        <div style="">
                <div class="row">
                    <div class="col-12">
                        <div class="m-2 mx-auto page-title-right d-block d-sm-none mx-sm-0 d-print-none">
                            
                        </div>
                        <div class="mt-0 mb-3 page-title-box">
                            <div class="page-title-right d-none d-sm-block d-print-none" style="margin-top:0px">
                                <div>
                                                                    </div>

                            </div>
                            <h1 class="mt-0 page-title mt-sm-3">TOPICAL PAST PAPER QUESTIONS
                                        <a target="_blank" href="https://www.youtube.com/watch?v=YGy46ekDyZE" tabindex="0" data-plugin="tippy" data-tippy-size="small" data-tippy-placement="bottom" data-tippy="" data-original-title="Video"><span class="fi fi-rr-play-circle fa-sm text-info ms-1 align-middle">
            </span></a>
                                </h1>
                            <p class="mb-0 page-description fs-6"> Topical past papers are an essential tool for any student looking to succeed in their exams. These questions are similar to regular past paper questions, but instead of covering all topics, they focus on a specific topic or theme. This allows students to test their knowledge in a more targeted way and identify any areas they may need to improve on. By practicing with Topical past papers, students have the opportunity to enhance their understanding of a topic and build their confidence ahead of the exam.</p>
                                                <a class="my-1 d-inline-block package-status-badge" href="https://www.exam-mate.com/topicalpastpapers/pricing">
    <span class="badge badge-soft-success d-block p-1"><span class="far fa-user pe-1 fa-lg"> </span> Subscribed till 27-May-2025 </span>
        
    </a>
                                    </div>
                    </div>
                </div>

            </div>
            <div id="app">
                
                    <div wire:id="k7239Q4eZPKgeoIAKl2h">
    <div class="" style="background:transparent">
        <div class="card card-body p-2 mb-1">
            <div class="filter-box-loading" wire:loading.delay="">
                <div class="spinner-border text-danger" role="status"></div>
            </div>
            <div id="filter-box">
                <div class="row no-gutters ps-2">
                    <div class="col-12 col-sm-auto ps-0 my-1 my-sm-0" x-init="$dispatch('filterResults')">
    <div class="dropdown">
    <button class="w-100 d-flex align-items-center btn btn-light dropdown-toggle text-left  filter-input filter-input-primary" type="button" id="" data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
        <span class="d-inline-block me-1">
            <span class="fi fi-rr-globe fa-lg"> </span>
        </span>
        <span class="d-inline-block w-100">
            <div class="d-flex justify-content-between">
                <span>
                    <span class="d-inline-block text-left pe-1" style="font-size:0.9em; text-align:left"> Curriculum: </span>
                    <span class="d-inline-block" style="font-weight:700;font-size:1.2em;vertical-align: middle;">

                        <span class="">
                                                        CIE IGCSE
                                                    </span>


                    </span>
                </span>
                <span class="ps-1 " style="font-weight:700;font-size:1.2em;vertical-align: middle;">
                    <i class="mdi mdi-chevron-down"></i>
                </span>
            </div>
        </span>

    </button>
    <div class="dropdown-menu p-0" style="max-height: 500px;min-width:300px;overflow-y: auto; overflow-x:hidden;" data-popper-placement="bottom-end" aria-labelledby="" wire:ignore="">
                <h5 class="mt-0 mb-0 px-2 py-1" style="color:#620a07;background: #f6fafc;  ">
                        <img src="https://www.exam-mate.com/storage/450817/Tick.webp" class="img-fluid me-1" alt="" style="width:15px;">
                        CIE </h5>
        <div class="p-1 row" style="padding-left: 2.5rem !important;">
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 3)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> IGCSE</a>
            </div>
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 5)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> A-LEVEL</a>
            </div>
                    </div>
                <h5 class="mt-0 mb-0 px-2 py-1" style="color:#620a07;background: #f6fafc;  ">
                        <img src="https://www.exam-mate.com/storage/450818/Tick.webp" class="img-fluid me-1" alt="" style="width:15px;">
                        Edexcel </h5>
        <div class="p-1 row" style="padding-left: 2.5rem !important;">
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 4)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> IGCSE</a>
            </div>
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 21)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> IGCSE (9-1)</a>
            </div>
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 6)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> A-LEVEL</a>
            </div>
                    </div>
                <h5 class="mt-0 mb-0 px-2 py-1" style="color:#620a07;background: #f6fafc;  ">
                        <img src="https://www.exam-mate.com/storage/450819/Tick.webp" class="img-fluid me-1" alt="" style="width:15px;">
                        IB </h5>
        <div class="p-1 row" style="padding-left: 2.5rem !important;">
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 7)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> DIPLOMA</a>
            </div>
                    </div>
                <h5 class="mt-0 mb-0 px-2 py-1" style="color:#620a07;background: #f6fafc;  ">
                        <img src="https://www.exam-mate.com/storage/450821/Tick.webp" class="img-fluid me-1" alt="" style="width:15px;">
                        OCR </h5>
        <div class="p-1 row" style="padding-left: 2.5rem !important;">
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 14)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> GCSE (9-1)</a>
            </div>
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 19)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> AS</a>
            </div>
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 16)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> A-LEVEL</a>
            </div>
                    </div>
                <h5 class="mt-0 mb-0 px-2 py-1" style="color:#620a07;background: #f6fafc;  ">
                        <img src="https://www.exam-mate.com/storage/450820/Tick.webp" class="img-fluid me-1" alt="" style="width:15px;">
                        AQA </h5>
        <div class="p-1 row" style="padding-left: 2.5rem !important;">
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 15)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> GCSE</a>
            </div>
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 18)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> AS</a>
            </div>
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 17)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> A-LEVEL</a>
            </div>
                    </div>
                <h5 class="mt-0 mb-0 px-2 py-1" style="color:#620a07;background: #f6fafc;  ">
                        <img src="https://www.exam-mate.com/storage/450822/Tick.webp" class="img-fluid me-1" alt="" style="width:15px;">
                        MALAYSIA </h5>
        <div class="p-1 row" style="padding-left: 2.5rem !important;">
                        <div class="col-sm-6 pb-1" wire:click="$set('categoryID', 13)" role="button">
                <a class="fw-bold me-1 cat-item" style=""> SPM</a>
            </div>
                    </div>
            </div>
</div>
</div>
<div class="col-12 col-sm-auto ps-0 my-1 my-sm-0">
    <div class="dropdown">
    <button class="w-100 d-flex align-items-center btn btn-light dropdown-toggle text-left  filter-input filter-input-primary" type="button" id="" data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
        <span class="d-inline-block me-1">
            <span class="fi fi-rr-books fa-lg"> </span>
        </span>
        <span class="d-inline-block w-100">
            <div class="d-flex justify-content-between">
                <span>
                    <span class="d-inline-block text-left pe-1" style="font-size:0.9em; text-align:left"> Subject: </span>
                    <span class="d-inline-block" style="font-weight:700;font-size:1.2em;vertical-align: middle;">
                    
                        <span class=""> 
            Biology(0610)</span>
                    </span>
                </span>
                <span class="ps-1 " style="font-weight:700;font-size:1.2em;vertical-align: middle;">
                    <i class="mdi mdi-chevron-down"></i>
                </span>
            </div>
        </span>
    </button>
    <div class="dropdown-menu" aria-labelledby="" style="max-height: 500px;overflow-y: auto;">
                <a class="dropdown-item" wire:click="$set('subjectID', 11)" role="button">
            <span class="text-dark fw-bold">
             Additional Mathematics(0606)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 247)" role="button">
            <span class="text-dark fw-bold">
             Biology 9-1(0970)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 13)" role="button">
            <span class="text-dark fw-bold">
             Biology(0610)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 249)" role="button">
            <span class="text-dark fw-bold">
             Chemistry 9-1(0971)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 16)" role="button">
            <span class="text-dark fw-bold">
             Chemistry(0620)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 250)" role="button">
            <span class="text-dark fw-bold">
             Computer Science 9-1(0984)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 128)" role="button">
            <span class="text-dark fw-bold">
             Computer Science(0478)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 252)" role="button">
            <span class="text-dark fw-bold">
             Economics 9-1(0987)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 17)" role="button">
            <span class="text-dark fw-bold">
             Economics(0455)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 18)" role="button">
            <span class="text-dark fw-bold">
             English 1st Language(0500)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 19)" role="button">
            <span class="text-dark fw-bold">
             English 2nd Language(0510)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 24)" role="button">
            <span class="text-dark fw-bold">
             History(0470)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 23)" role="button">
            <span class="text-dark fw-bold">
             ICT(0417)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 262)" role="button">
            <span class="text-dark fw-bold">
             Mathematics 9-1(0980)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 136)" role="button">
            <span class="text-dark fw-bold">
             Mathematics International(0607)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 22)" role="button">
            <span class="text-dark fw-bold">
             Mathematics(0580)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 265)" role="button">
            <span class="text-dark fw-bold">
             Physics 9-1(0972)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 29)" role="button">
            <span class="text-dark fw-bold">
             Physics(0625)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 279)" role="button">
            <span class="text-dark fw-bold">
             Religious Studies(0490)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 25)" role="button">
            <span class="text-dark fw-bold">
             Science Combined(0653)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 26)" role="button">
            <span class="text-dark fw-bold">
             Science Coordinate(0654)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 266)" role="button">
            <span class="text-dark fw-bold">
             Sciences - Co-ordinated 9-1(0973)</span>
        </a>
                <a class="dropdown-item" wire:click="$set('subjectID', 30)" role="button">
            <span class="text-dark fw-bold">
             Sociology(0495)</span>
        </a>
            </div>
</div>
</div>
<div class="col-12 col-sm-auto ps-sm-0 ps-2 px-0 mx-auto mx-sm-0 me-1 align-middle pt-1 " style="font-size:1em">
    <span class="pd-flex align-items-center">
        <span class="badge bg-primary text-white">
                        Chapterized Till : Oct 2024
                    </span>
    </span>
</div>
                </div>
                <div class="extra-filter-box mt-1">
                    <div class="row gx-0">
                        <div class="col-md-11">
                            <div class="row gx-0">
                                <div class="col-md-4 p-2 extra-filter-input">
    <div class="dropdown ms-2">
     <button style="text-align:left;border:none;box-shadow:none;" class="w-100 d-flex align-items-center btn btn-light dropdown-toggle text-left filter-input bg-transparent p-0" type="button" id="" data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
         <span class="d-inline-block" style="flex-grow: 1;">
             <span class="d-block text-left" style="font-size:0.9em; text-align:left"> Topic(s): </span>
             <span class="d-flex justify-content-between">
                                      <span style="font-weight:700;font-size:1em;" id="topics-selected">CHARACTERISTICS AND ...</span>
                                  <i class="mdi mdi-chevron-down"></i>
             </span>
         </span>

     </button>
     <div class="dropdown-menu w-100" aria-labelledby="" style="max-height: 500px;overflow-y: auto;min-width:300px">
                      <div class="p-2 filter-menu-dropdown">
                                                       <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="CHARACTERISTICS AND CLASSIFICATION OF LIVING ORGANISMS" value="33" id="topic-33">
                         
                         <label class="form-check-label text-dark" for="topic-33">
                                                              CH1 -
                                                          CHARACTERISTICS AND CLASSIFICATION OF LIVING ORGANISMS
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="ORGANIZATION AND MAINTENANCE OF THE ORGANISM" value="34" id="topic-34">
                         
                         <label class="form-check-label text-dark" for="topic-34">
                                                              CH2 -
                                                          ORGANIZATION AND MAINTENANCE OF THE ORGANISM
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="MOVEMENT IN AND OUT OF CELLS" value="35" id="topic-35">
                         
                         <label class="form-check-label text-dark" for="topic-35">
                                                              CH3 -
                                                          MOVEMENT IN AND OUT OF CELLS
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="BIOLOGICAL MOLECULES" value="36" id="topic-36">
                         
                         <label class="form-check-label text-dark" for="topic-36">
                                                              CH4 -
                                                          BIOLOGICAL MOLECULES
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="ENZYMES" value="37" id="topic-37">
                         
                         <label class="form-check-label text-dark" for="topic-37">
                                                              CH5 -
                                                          ENZYMES
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="PLANT NUTRITION" value="38" id="topic-38">
                         
                         <label class="form-check-label text-dark" for="topic-38">
                                                              CH6 -
                                                          PLANT NUTRITION
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="HUMAN NUTRITION" value="39" id="topic-39">
                         
                         <label class="form-check-label text-dark" for="topic-39">
                                                              CH7 -
                                                          HUMAN NUTRITION
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="TRANSPORT IN PLANTS" value="40" id="topic-40">
                         
                         <label class="form-check-label text-dark" for="topic-40">
                                                              CH8 -
                                                          TRANSPORT IN PLANTS
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="TRANSPORT IN ANIMALS" value="41" id="topic-41">
                         
                         <label class="form-check-label text-dark" for="topic-41">
                                                              CH9 -
                                                          TRANSPORT IN ANIMALS
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="DISEASES AND IMMUNITY" value="42" id="topic-42">
                         
                         <label class="form-check-label text-dark" for="topic-42">
                                                              CH10 -
                                                          DISEASES AND IMMUNITY
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="GAS EXCHANGE IN HUMANS" value="43" id="topic-43">
                         
                         <label class="form-check-label text-dark" for="topic-43">
                                                              CH11 -
                                                          GAS EXCHANGE IN HUMANS
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="RESPIRATION" value="44" id="topic-44">
                         
                         <label class="form-check-label text-dark" for="topic-44">
                                                              CH12 -
                                                          RESPIRATION
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="EXCRETION IN HUMANS" value="45" id="topic-45">
                         
                         <label class="form-check-label text-dark" for="topic-45">
                                                              CH13 -
                                                          EXCRETION IN HUMANS
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="CO-ORDINATION AND RESPONSE" value="46" id="topic-46">
                         
                         <label class="form-check-label text-dark" for="topic-46">
                                                              CH14 -
                                                          CO-ORDINATION AND RESPONSE
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="DRUGS" value="47" id="topic-47">
                         
                         <label class="form-check-label text-dark" for="topic-47">
                                                              CH15 -
                                                          DRUGS
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="REPRODUCTION" value="48" id="topic-48">
                         
                         <label class="form-check-label text-dark" for="topic-48">
                                                              CH16 -
                                                          REPRODUCTION
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="INHERITANCE" value="49" id="topic-49">
                         
                         <label class="form-check-label text-dark" for="topic-49">
                                                              CH17 -
                                                          INHERITANCE
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="VARIATION AND SELECTION" value="50" id="topic-50">
                         
                         <label class="form-check-label text-dark" for="topic-50">
                                                              CH18 -
                                                          VARIATION AND SELECTION
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="ORGANISMS AND THEIR ENVIRONMENT" value="51" id="topic-51">
                         
                         <label class="form-check-label text-dark" for="topic-51">
                                                              CH19 -
                                                          ORGANISMS AND THEIR ENVIRONMENT
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="BIOTECHNOLOGY AND GENETIC ENGINEERING" value="52" id="topic-52">
                         
                         <label class="form-check-label text-dark" for="topic-52">
                                                              CH20 -
                                                          BIOTECHNOLOGY AND GENETIC ENGINEERING
                         </label>
                     </div>
                                      <div class="form-check mb-1">
                                                      <input type="checkbox" name="topics[]" class="form-check-input" wire:model.defer="topicIDs" data-name="HUMAN INFLUENCES ON ECOSYSTEMS" value="53" id="topic-53">
                         
                         <label class="form-check-label text-dark" for="topic-53">
                                                              CH21 -
                                                          HUMAN INFLUENCES ON ECOSYSTEMS
                         </label>
                     </div>
                              </div>
              </div>
 </div>

 </div>
<div class="col-md-2 p-2 extra-filter-input">
    <div class="dropdown ms-2">
     <button style="text-align:left;border:none;box-shadow:none;" class="p-0 text-left bg-transparent w-100 d-flex align-items-center btn btn-light dropdown-toggle filter-input" type="button" id="" data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
         <span class="d-inline-block" style="flex-grow: 1;">
             <span class="text-left d-block" style="font-size:0.9em; text-align:left">
                                      Paper(s):
                              </span>
             <span class="d-flex justify-content-between">
                                      <span style="font-weight:700;font-size:1em;" id="papers-selected" data-name="Select Paper">2 Selected</span>
                                  <i class="mdi mdi-chevron-down"></i>
             </span>
         </span>

     </button>
     <div class="dropdown-menu w-100" aria-labelledby="">
         <div class="p-2 filter-menu-dropdown">
                                               <div class="mb-1 form-check">
                                              <input type="checkbox" name="papers[]" class="form-check-input" wire:model.defer="paperIDs" data-name="1 (Core)" value="4" id="paper-4">
                     
                     <label class="form-check-label text-dark" for="paper-4">1 (Core)
                     </label>
                 </div>
                                               <div class="mb-1 form-check">
                                              <input type="checkbox" name="papers[]" class="form-check-input" wire:model.defer="paperIDs" data-name="2 (Extended)" value="6" id="paper-6">
                     
                     <label class="form-check-label text-dark" for="paper-6">2 (Extended)
                     </label>
                 </div>
                                               <div class="mb-1 form-check">
                                              <input type="checkbox" name="papers[]" class="form-check-input" wire:model.defer="paperIDs" data-name="3 (Core)" value="5" id="paper-5">
                     
                     <label class="form-check-label text-dark" for="paper-5">3 (Core)
                     </label>
                 </div>
                                               <div class="mb-1 form-check">
                                              <input type="checkbox" name="papers[]" class="form-check-input" wire:model.defer="paperIDs" data-name="4 (Extended)" value="7" id="paper-7">
                     
                     <label class="form-check-label text-dark" for="paper-7">4 (Extended)
                     </label>
                 </div>
                                               <div class="mb-1 form-check">
                                              <input type="checkbox" name="papers[]" class="form-check-input" wire:model.defer="paperIDs" data-name="5" value="8" id="paper-8">
                     
                     <label class="form-check-label text-dark" for="paper-8">5
                     </label>
                 </div>
                                               <div class="mb-1 form-check">
                                              <input type="checkbox" name="papers[]" class="form-check-input" wire:model.defer="paperIDs" data-name="6" value="9" id="paper-9">
                     
                     <label class="form-check-label text-dark" for="paper-9">6
                     </label>
                 </div>
                      </div>
     </div>
 </div>



 </div>
<div class="col-md-2 p-2 extra-filter-input">
    <div class="dropdown ms-2">

     <button style="text-align:left;border:none;box-shadow:none;" class="w-100 d-flex align-items-center btn btn-light dropdown-toggle text-left filter-input bg-transparent p-0" type="button" id="" data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
         <span class="d-inline-block" style="flex-grow: 1;">
             <span class="d-block text-left" style="font-size:0.9em; text-align:left"> Year(s): </span>
             <span class="d-flex justify-content-between">
                 <span style="font-weight:700;font-size:1em;" id="years-selected">10 Selected</span>
                 <i class="mdi mdi-chevron-down"></i> </span>
         </span>

     </button>
     <div class="dropdown-menu w-100" aria-labelledby="">
         <div class="p-2 filter-menu-dropdown">
                                           <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2024" id="year-2024">
                                          

                     <label class="form-check-label text-dark" for="year-2024">2024</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2023" id="year-2023">
                                          

                     <label class="form-check-label text-dark" for="year-2023">2023</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2022" id="year-2022">
                                          

                     <label class="form-check-label text-dark" for="year-2022">2022</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2021" id="year-2021">
                                          

                     <label class="form-check-label text-dark" for="year-2021">2021</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2020" id="year-2020">
                                          

                     <label class="form-check-label text-dark" for="year-2020">2020</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2019" id="year-2019">
                                          

                     <label class="form-check-label text-dark" for="year-2019">2019</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2018" id="year-2018">
                                          

                     <label class="form-check-label text-dark" for="year-2018">2018</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2017" id="year-2017">
                                          

                     <label class="form-check-label text-dark" for="year-2017">2017</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2016" id="year-2016">
                                          

                     <label class="form-check-label text-dark" for="year-2016">2016</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2015" id="year-2015">
                                          

                     <label class="form-check-label text-dark" for="year-2015">2015</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2014" id="year-2014">
                                          

                     <label class="form-check-label text-dark" for="year-2014">2014</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2013" id="year-2013">
                                          

                     <label class="form-check-label text-dark" for="year-2013">2013</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2012" id="year-2012">
                                          

                     <label class="form-check-label text-dark" for="year-2012">2012</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2011" id="year-2011">
                                          

                     <label class="form-check-label text-dark" for="year-2011">2011</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2010" id="year-2010">
                                          

                     <label class="form-check-label text-dark" for="year-2010">2010</label>
                                      </div>
                                               <div class="form-check mb-1">
                                          <input type="checkbox" name="years[]" class="form-check-input" wire:model.defer="years" value="2009" id="year-2009">
                                          

                     <label class="form-check-label text-dark" for="year-2009">2009</label>
                                      </div>
                              </div>
         </div>
     </div>

     </div>
<div class="col-md-2 p-2 extra-filter-input">
    <div class="dropdown ms-2">
     <button style="text-align:left;border:none;box-shadow:none;" class="w-100 d-flex align-items-center btn btn-light dropdown-toggle text-left filter-input bg-transparent p-0" type="button" id="" data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
         <span class="d-inline-block" style="flex-grow: 1;">
             <span class="d-block text-left" style="font-size:0.9em; text-align:left"> Season(s): </span>
             <span class="d-flex justify-content-between">
                              <span style="font-weight:700;font-size:1em;" id="seasons-selected">3 Selected</span>
                          <i class="mdi mdi-chevron-down"></i> </span>
         </span>

     </button>
     <div class="dropdown-menu w-100" aria-labelledby="">
         <div class="p-2 filter-menu-dropdown">
                          <div class="form-check mb-1">
                                  <input type="checkbox" name="seasons[]" class="form-check-input" wire:model.defer="seasons" data-name="Summer" value="SUMMER" id="season-SUMMER">
                                  <label class="form-check-label text-dark" for="season-SUMMER">Summer </label>
             </div>
                          <div class="form-check mb-1">
                                  <input type="checkbox" name="seasons[]" class="form-check-input" wire:model.defer="seasons" data-name="Winter" value="WINTER" id="season-WINTER">
                                  <label class="form-check-label text-dark" for="season-WINTER">Winter </label>
             </div>
                          <div class="form-check mb-1">
                                  <input type="checkbox" name="seasons[]" class="form-check-input" wire:model.defer="seasons" data-name="Spring" value="SPRING" id="season-SPRING">
                                  <label class="form-check-label text-dark" for="season-SPRING">Spring </label>
             </div>
                      </div>
     </div>
 </div>


 </div>


<div class="col-md-2  p-2 extra-filter-input">
    <div class="dropdown ms-2">
     <button style="text-align:left;border:none;box-shadow:none;" class="w-100 d-flex align-items-center btn btn-light dropdown-toggle text-left filter-input bg-transparent p-0" type="button" id="" data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
         <span class="d-inline-block" style="flex-grow: 1;">
             <span class="d-block text-left" style="font-size:0.9em; text-align:left"> Zone(s): </span>
             <span class="d-flex justify-content-between">
                              <span style="font-weight:700;font-size:1em;" id="zones-selected">3 Selected</span>
                          <i class="mdi mdi-chevron-down"></i> </span>
         </span>

     </button>
     <div class="dropdown-menu w-100" aria-labelledby="">
         <div class="p-2 filter-menu-dropdown">
                          <div class="form-check mb-1">
                             <input type="checkbox" name="zones[]" class="form-check-input" wire:model.defer="zoneIDs" data-name="1" value="1" id="zone-1">
                                 <label class="form-check-label text-dark" for="zone-1">1 </label>
             </div>
                          <div class="form-check mb-1">
                             <input type="checkbox" name="zones[]" class="form-check-input" wire:model.defer="zoneIDs" data-name="2" value="2" id="zone-2">
                                 <label class="form-check-label text-dark" for="zone-2">2 </label>
             </div>
                          <div class="form-check mb-1">
                             <input type="checkbox" name="zones[]" class="form-check-input" wire:model.defer="zoneIDs" data-name="3" value="3" id="zone-3">
                                 <label class="form-check-label text-dark" for="zone-3">3 </label>
             </div>
                      </div>
     </div>
 </div>

 </div>
                            </div>
                        </div>
                        <div class="col-md-1">
                            <button class="btn btn-primary waves-effect waves-light btn-lg w-100 filter-button d-none d-md-block" style="height:60px" wire:click="filter()">
                                <i class="fi fi-br-search fa-lg"></i>
                            </button>
                        </div>
                    </div>
                </div>

                <button class="mt-2 btn btn-primary waves-effect waves-light btn-lg w-100 filter-button d-block d-md-none" style="height:60px" wire:click="filter()">
                    <i class="fi fi-br-search fa-lg"></i>
                </button>
            </div>
        </div>
    </div>
</div>

<!-- Livewire Component wire-end:k7239Q4eZPKgeoIAKl2h -->    <div class="row">
        <div class="col-xl-3 col-lg-4 pe-lg-0">
            <div wire:id="pBHdAv5dWkx7jyj6fI30" class="pt-0 card card-body" style="padding:0.1rem">
    <div class="qpreview-box-loading" wire:loading.delay="">
        <div class="spinner-border text-danger" role="status"></div>
    </div>
            <ul class="nav nav-tabs tabs-no-hover-border justify-content-center" style="border-bottom:none;position:relative">
        <li class="nav-item">
        <a href="javascript:void(0);" class="nav-link nav-no-border" wire:click="toggleOrder()">
                        <i class="fi fi-br-sort-amount-up  d-block text-center"></i>
        <span style="font-size:0.6rem; font-weight:700"> Asc</span>
            </a>
</li>
        <li>
    <a href="javascript:void(0);" class="nav-link nav-no-border nav-no-hover pe-none">
        <i class="fi fi-br-filter  d-block text-center" style="position:relative">
        </i>
        <span style="font-size:0.6rem; font-weight:700" class="text-center badge badge-soft-secondary rounded-pill">
            219
        </span>

    </a>
</li>
        <li>
    <a href="javascript:void(0);" class="nav-link nav-no-border nav-no-hover pe-none">
        <i class="fi fi-br-eye  d-block text-center" style="position:relative">
        </i>
        <span style="font-size:0.6rem; font-weight:700" class="text-center badge badge-soft-secondary rounded-pill">
            1 - 25
        </span>

    </a>
</li>
    
        <li class="nav-item dropdown randomdropdown" style="position:relative">
        <a href="javascript:void(0);" data-popper-placement="bottom-start" class="bs-tooltip-end dropstart nav-link nav-no-border " data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
            <i class="fi fi-br-shuffle  d-block text-center"></i>
            <span style="font-size:0.6rem; font-weight:700"> Random </span>
        </a>
        <div class="dropdown-menu" style="left:-300px !important;">
            <div class="px-4 py-3" style="min-width:300px;">
                <div class="mb-3 mt-1">
                    <div class="form-check form-switch mt-2 form-check-dark">
                        <input type="checkbox" id="is_random" class="form-check-input" wire:model.defer="random">
                        <label class="form-check-label text-dark" for="random_select">Random Questions</label>
                    </div>
                </div>

                <div class="mb-3 mt-2">
                    <label for="num_qs" class="form-label">Number of Questions</label>
                    <input type="number" class="form-control" min="1" max="75" id="num_qs" wire:model.defer="randomNumQuestions">
                    <span class="help-block"><small>Only applied when random questions is enabled. <strong>Max: 75</strong></small></span>
                </div>

                <button class="copy-link btn btn-primary mt-1" type="button" wire:click="randomSelect">Update</button>

                                                            </div>
        </div>
    </li>
    
</ul>
    
    <div class="question-list">
        <div style="height:calc(80vh - 45px);overflow-y: auto;" class="mb-auto">
            <ul class="list-group results-box" id="questions-list1" x-init="selectFirstQuestion(1)">
                                    <li id="qid-179156-1" wire:click="$emit('questionSelected',179156,1)" onclick="javascript:selectQuestion(179156,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/474241\/0610_w24_qp_43_Q2_Part-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/474242\/0610_w24_qp_43_Q2_Part-2.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/474243\/0610_w24_qp_43_Q2_Part-3.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/474244\/0610_w24_qp_43_Q2_Part-4.png&quot;],&quot;answer_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/474256\/0610_w24_ms_43_Q2_1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/474257\/0610_w24_ms_43_Q2_2.png&quot;],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:null,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms, Plant Nutrition, Transport In Plants&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center active">
    <span>
        0610/43_Winter_2024_Q2
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="X3yO8Lv2VfaoFSP0ZmqU">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(179156)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:X3yO8Lv2VfaoFSP0ZmqU -->        </span>
                    </span>
</li>
                                    <li id="qid-179149-1" wire:click="$emit('questionSelected',179149,1)" onclick="javascript:selectQuestion(179149,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/474217\/0610_w24_qp_42_Q1_Part-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/474218\/0610_w24_qp_42_Q1_Part-2.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/474219\/0610_w24_qp_42_Q1_Part-3.png&quot;],&quot;answer_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/474231\/0610_w24_ms_42_Q1_1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/474232\/0610_w24_ms_42_Q1_2.png&quot;],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:null,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms, Biological Molecules, Human Nutrition&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/42_Winter_2024_Q1
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="iyMyScLLvjxUwoCwN2IP">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(179149)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:iyMyScLLvjxUwoCwN2IP -->        </span>
                    </span>
</li>
                                    <li id="qid-179145-1" wire:click="$emit('questionSelected',179145,1)" onclick="javascript:selectQuestion(179145,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/474200\/0610_w24_qp_41_Q3_Part-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/474201\/0610_w24_qp_41_Q3_Part-2.png&quot;],&quot;answer_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/474212\/0610_w24_ms_41_Q3_1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/474213\/0610_w24_ms_41_Q3_2.png&quot;],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:null,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms, Diseases And Immunity&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/41_Winter_2024_Q3
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="y4Q6euwUIw7gHeZ2lbKe">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(179145)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:y4Q6euwUIw7gHeZ2lbKe -->        </span>
                    </span>
</li>
                                    <li id="qid-179064-1" wire:click="$emit('questionSelected',179064,1)" onclick="javascript:selectQuestion(179064,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/474016\/0610_w24_qp_23_Q4.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;C&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/23_Winter_2024_Q4
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="2aokMpXRtIckJ2aDmfHj">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(179064)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:2aokMpXRtIckJ2aDmfHj -->        </span>
                    </span>
</li>
                                    <li id="qid-179063-1" wire:click="$emit('questionSelected',179063,1)" onclick="javascript:selectQuestion(179063,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/474015\/0610_w24_qp_23_Q3.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;A&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/23_Winter_2024_Q3
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="JWOwEHPKgoXNmGPmpUd8">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(179063)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:JWOwEHPKgoXNmGPmpUd8 -->        </span>
                    </span>
</li>
                                    <li id="qid-179062-1" wire:click="$emit('questionSelected',179062,1)" onclick="javascript:selectQuestion(179062,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/474014\/0610_w24_qp_23_Q2.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;D&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/23_Winter_2024_Q2
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="XiOHrL5a1lmOcdB8sAVu">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(179062)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:XiOHrL5a1lmOcdB8sAVu -->        </span>
                    </span>
</li>
                                    <li id="qid-179019-1" wire:click="$emit('questionSelected',179019,1)" onclick="javascript:selectQuestion(179019,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/473958\/0610_w24_qp_22_Q3.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;D&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/22_Winter_2024_Q3
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="OXfRz45Lbz6OH4R1h9wq">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(179019)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:OXfRz45Lbz6OH4R1h9wq -->        </span>
                    </span>
</li>
                                    <li id="qid-179018-1" wire:click="$emit('questionSelected',179018,1)" onclick="javascript:selectQuestion(179018,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/473957\/0610_w24_qp_22_Q2.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;B&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/22_Winter_2024_Q2
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="3aD7DSjUQwUolVhPjK6I">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(179018)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:3aD7DSjUQwUolVhPjK6I -->        </span>
                    </span>
</li>
                                    <li id="qid-179017-1" wire:click="$emit('questionSelected',179017,1)" onclick="javascript:selectQuestion(179017,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/473956\/0610_w24_qp_22_Q1.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;A&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/22_Winter_2024_Q1
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="pyDnUMNGCg5FLAAF0YqO">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(179017)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:pyDnUMNGCg5FLAAF0YqO -->        </span>
                    </span>
</li>
                                    <li id="qid-178970-1" wire:click="$emit('questionSelected',178970,1)" onclick="javascript:selectQuestion(178970,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/473879\/0610_w24_qp_21_Q4.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;A&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/21_Winter_2024_Q4
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="MLXC6c3rfYai2JA8qiKq">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(178970)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:MLXC6c3rfYai2JA8qiKq -->        </span>
                    </span>
</li>
                                    <li id="qid-178969-1" wire:click="$emit('questionSelected',178969,1)" onclick="javascript:selectQuestion(178969,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/473878\/0610_w24_qp_21_Q3.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;B&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/21_Winter_2024_Q3
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="IIViM5JofiPcjgRoZcNj">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(178969)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:IIViM5JofiPcjgRoZcNj -->        </span>
                    </span>
</li>
                                    <li id="qid-178968-1" wire:click="$emit('questionSelected',178968,1)" onclick="javascript:selectQuestion(178968,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/473877\/0610_w24_qp_21_Q2.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;C&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/21_Winter_2024_Q2
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="5137w6vEvQWSZt1lFRvS">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(178968)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:5137w6vEvQWSZt1lFRvS -->        </span>
                    </span>
</li>
                                    <li id="qid-178967-1" wire:click="$emit('questionSelected',178967,1)" onclick="javascript:selectQuestion(178967,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/473876\/0610_w24_qp_21_Q1.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;C&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/21_Winter_2024_Q1
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="KO5s19tD02KU0x99dneM">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(178967)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:KO5s19tD02KU0x99dneM -->        </span>
                    </span>
</li>
                                    <li id="qid-170280-1" wire:click="$emit('questionSelected',170280,1)" onclick="javascript:selectQuestion(170280,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453588\/0610_s24_qp_43_Q6-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453589\/0610_s24_qp_43_Q6-2.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453590\/0610_s24_qp_43_Q6-3.png&quot;],&quot;answer_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453600\/0610_s24_ms_43_MS6-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453601\/0610_s24_ms_43_MS6-2.png&quot;],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:null,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms, Human Influences On Ecosystems&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/43_Summer_2024_Q6
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="cydWHCZBVDM3WDWbLu58">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170280)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:cydWHCZBVDM3WDWbLu58 -->        </span>
                    </span>
</li>
                                    <li id="qid-170276-1" wire:click="$emit('questionSelected',170276,1)" onclick="javascript:selectQuestion(170276,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453577\/0610_s24_qp_43_Q2-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453578\/0610_s24_qp_43_Q2-2.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453579\/0610_s24_qp_43_Q2-3.png&quot;],&quot;answer_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453592\/0610_s24_ms_43_MS2-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453593\/0610_s24_ms_43_MS2-2.png&quot;],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:null,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms, Reproduction&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/43_Summer_2024_Q2
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="o8xx5bhhWBs7NOO4mx4b">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170276)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:o8xx5bhhWBs7NOO4mx4b -->        </span>
                    </span>
</li>
                                    <li id="qid-170273-1" wire:click="$emit('questionSelected',170273,1)" onclick="javascript:selectQuestion(170273,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453560\/0610_s24_qp_42_Q5-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453561\/0610_s24_qp_42_Q5-2.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453562\/0610_s24_qp_42_Q5-3.png&quot;],&quot;answer_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453571\/0610_s24_ms_42_MS5-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453572\/0610_s24_ms_42_MS5-2.png&quot;],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:null,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms, Reproduction&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/42_Summer_2024_Q5
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="mGXIrKuRk4XBtV336A8O">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170273)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:mGXIrKuRk4XBtV336A8O -->        </span>
                    </span>
</li>
                                    <li id="qid-170269-1" wire:click="$emit('questionSelected',170269,1)" onclick="javascript:selectQuestion(170269,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453524\/0610_s24_qp_42_Q1-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453525\/0610_s24_qp_42_Q1-2.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453526\/0610_s24_qp_42_Q1-3.png&quot;],&quot;answer_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453565\/0610_s24_ms_42_MS1.png&quot;],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:null,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms, Respiration, Biotechnology And Genetic Engineering&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/42_Summer_2024_Q1
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="X2WCwsYdqx58c94ER2J3">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170269)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:X2WCwsYdqx58c94ER2J3 -->        </span>
                    </span>
</li>
                                    <li id="qid-170243-1" wire:click="$emit('questionSelected',170243,1)" onclick="javascript:selectQuestion(170243,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453474\/0610_s24_qp_41_Q1-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453475\/0610_s24_qp_41_Q1-2.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453476\/0610_s24_qp_41_Q1-3.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453477\/0610_s24_qp_41_Q1-4.png&quot;],&quot;answer_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453508\/0610_s24_ms_41_MS1-1.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453509\/0610_s24_ms_41_MS1-2.png&quot;,&quot;https:\/\/www.exam-mate.com\/media\/questions\/453510\/0610_s24_ms_41_MS1-3.png&quot;],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:null,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms, Transport In Animals&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/41_Summer_2024_Q1
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="wqCHQs3imy1qjQfhsvsd">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170243)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:wqCHQs3imy1qjQfhsvsd -->        </span>
                    </span>
</li>
                                    <li id="qid-170151-1" wire:click="$emit('questionSelected',170151,1)" onclick="javascript:selectQuestion(170151,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453293\/0610_s24_qp_23_Q2.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;C&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/23_Summer_2024_Q2
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="HfOhgKq1aoF7OcdRT4tv">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170151)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:HfOhgKq1aoF7OcdRT4tv -->        </span>
                    </span>
</li>
                                    <li id="qid-170150-1" wire:click="$emit('questionSelected',170150,1)" onclick="javascript:selectQuestion(170150,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453292\/0610_s24_qp_23_Q1.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;B&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/23_Summer_2024_Q1
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="AyrUY4yKUmgB5ykuEso7">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170150)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:AyrUY4yKUmgB5ykuEso7 -->        </span>
                    </span>
</li>
                                    <li id="qid-170101-1" wire:click="$emit('questionSelected',170101,1)" onclick="javascript:selectQuestion(170101,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453223\/0610_s24_qp_22_Q2.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;A&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/22_Summer_2024_Q2
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="iPFmDiQZPeJM4OcaJbtl">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170101)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:iPFmDiQZPeJM4OcaJbtl -->        </span>
                    </span>
</li>
                                    <li id="qid-170098-1" wire:click="$emit('questionSelected',170098,1)" onclick="javascript:selectQuestion(170098,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453220\/0610_s24_qp_22_Q1.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;A&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/22_Summer_2024_Q1
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="WZuSjfD8dSFWZXcsppgy">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170098)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:WZuSjfD8dSFWZXcsppgy -->        </span>
                    </span>
</li>
                                    <li id="qid-170031-1" wire:click="$emit('questionSelected',170031,1)" onclick="javascript:selectQuestion(170031,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453130\/0610_s24_qp_21_Q2.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;B&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/21_Summer_2024_Q2
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="3aIMSvEagBAQLDzujft4">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170031)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:3aIMSvEagBAQLDzujft4 -->        </span>
                    </span>
</li>
                                    <li id="qid-170028-1" wire:click="$emit('questionSelected',170028,1)" onclick="javascript:selectQuestion(170028,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/453127\/0610_s24_qp_21_Q1.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;D&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/21_Summer_2024_Q1
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="NQY3KRucKueaDukhmeif">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(170028)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:NQY3KRucKueaDukhmeif -->        </span>
                    </span>
</li>
                                    <li id="qid-181122-1" wire:click="$emit('questionSelected',181122,1)" onclick="javascript:selectQuestion(181122,1,'{&quot;question_images&quot;:[&quot;https:\/\/www.exam-mate.com\/media\/questions\/478411\/0610_m24_qp_22_Q4.png&quot;],&quot;answer_images&quot;:[],&quot;mcq_answer&quot;:&quot;A&quot;,&quot;question_text&quot;:null,&quot;answer_text&quot;:&quot;C&quot;,&quot;topics&quot;:&quot;Characteristics And Classification Of Living Organisms&quot;}')" role="button" class="question-item-1 list-group-item d-flex justify-content-between align-items-center">
    <span>
        0610/22_Spring_2024_Q4
    </span>
    <span>
                        <span class="d-inline-block pe-1">
            <div wire:id="s4DAPym9zZ7NZWWwzeRi">
    <a class="text-dark" href="javascript:void(0);" wire:click="toggleFavorite(181122)" onclick="event.stopPropagation();">
                <i class="far fa-heart text-dark"></i>
            </a>
    </div>
        

<!-- Livewire Component wire-end:s4DAPym9zZ7NZWWwzeRi -->        </span>
                    </span>
</li>
                            </ul>
        </div>
        <div class="pt-1" style="min-height:45px;max-height:45px;">
                            <nav>
        <ul class="pagination justify-content-center pagination-sm">
            
                            <li class="page-item disabled" aria-disabled="true" aria-label="« Previous">
                    <span class="page-link" aria-hidden="true">‹</span>
                </li>
            
            
                            
                
                
                                                                                        <li class="page-item active" aria-current="page"><span class="page-link">1</span></li>
                                                                                                <li class="page-item"><a class="page-link" role="button" wire:click="gotoPage(2)">2</a></li>
                                                                                                <li class="page-item"><a class="page-link" role="button" wire:click="gotoPage(3)">3</a></li>
                                                                                                <li class="page-item"><a class="page-link" role="button" wire:click="gotoPage(4)">4</a></li>
                                                                                        
                                    <li class="page-item disabled" aria-disabled="true"><span class="page-link">...</span></li>
                
                
                                            
                
                
                                                                                        <li class="page-item"><a class="page-link" role="button" wire:click="gotoPage(8)">8</a></li>
                                                                                                <li class="page-item"><a class="page-link" role="button" wire:click="gotoPage(9)">9</a></li>
                                                                        
            
                            <li class="page-item">
                    <a class="page-link" role="button" wire:click="nextPage" rel="next" aria-label="Next »">›</a>
                </li>
                    </ul>
    </nav>

                    </div>
    </div>

</div>

<!-- Livewire Component wire-end:pBHdAv5dWkx7jyj6fI30 -->        </div>
        <div class="col-xl-9 col-lg-8 ps-lg-1">
            <div wire:id="TkCPbiaDZKlLcbdI4QuM">
    <div class="pt-0 card card-body" style="padding:0.1rem">
        <div class="row">
            <div class="col-md-12">
                <ul class="nav nav-tabs justify-content-center" style="border:none">
                                                                                                    <li class="nav-item dropdown">
                                <a href="#" class="nav-link nav-item nav-no-border dropdown-toggle" data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                    <i class="text-center fi fi-br-plus d-block"></i>
                                    <span style="font-size:0.6rem; font-weight:700"> Add to</span>
                                </a>
                                <div class="dropdown-menu" aria-labelledby="dropdownMenuLink">
                                    <a class="dropdown-item" href="#" wire:loading.attr="disabled" wire:click="$emit('openAddToExamsModal')"><strong>Exam</strong></a>
                                    <a class="dropdown-item" href="#" wire:loading.attr="disabled" wire:click="$emit('openAddToQuestionListModal')"><strong>Question List</strong></a>
                                </div>

                                <div wire:id="3QKwM23b9vNoFKuml5LA">
        <!-- Modal -->
<div x-data="{
        show: window.Livewire.find('3QKwM23b9vNoFKuml5LA').entangle('AddToListModalOpen').defer,
    }" x-init="() => {

        let el = document.querySelector('#modal-id-d7f54be00042f5ed1f36dec9335c93ad')

        let modal = new bootstrap.Modal(el);

        $watch('show', value => {
            if (value) {
                modal.show()
            } else {
                modal.hide()
            }
        });

        el.addEventListener('hide.bs.modal', function (event) {
          show = false
        })
    }" wire:ignore.self="" class="modal fade" tabindex="-1" id="modal-id-d7f54be00042f5ed1f36dec9335c93ad" aria-labelledby="modal-id-d7f54be00042f5ed1f36dec9335c93ad" aria-hidden="true" x-ref="modal-id-d7f54be00042f5ed1f36dec9335c93ad">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <h5 class="modal-title"></h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
            <div class="text-center mb-3">
                <span class="fi fi-br-bulb-nib fa-2x mb-2 d-inline-block text-primary"> </span>
                <h4 class="modal-title fs-2 fw-light">                     Add Question to Question Lists
                    </h4>
            </div>
            <div class="p-2 pt-0">
                <ul class="nav nav-tabs nav-pills1 navtab-bg1 nav-bordered1 justify-content-center show-exam-menu" wire:ignore="add-exam-button">
                    <li class="nav-item">
                        <a href="#existing-list-tab" data-bs-toggle="tab" aria-expanded="true" class="buildexam-topnav-item nav-link active nav-white">
                            To Existing List
                        </a>
                    </li>
                    <li class="nav-item">
                        <a href="#new-list-tab" data-bs-toggle="tab" aria-expanded="false" class="buildexam-topnav-item nav-link nav-white">
                            Create New List
                        </a>
                    </li>
                </ul>
                <div class="tab-content">
                    <div class="tab-pane show active" id="existing-list-tab" wire:ignore.self="add-exam-button" wire:key="addToExamsModal">
                        <ul class="list-group">
                                                        <li class="list-group-item d-flex justify-content-between align-items-center py-1">
                                <span class="default-font text-dark fw-bold">Question List 2
                                    <span role="button" class="badge badge-soft-dark-blue rounded-pill">
                                        1 / 300
                                    </span>
                                </span>
                                                                                                <span role="button" class="badge badge-soft-success rounded-pill" wire:click="addQuestion(152239)">Add</span>
                                                                                            </li>
                                                        <li class="list-group-item d-flex justify-content-between align-items-center py-1">
                                <span class="default-font text-dark fw-bold">Question List 1
                                    <span role="button" class="badge badge-soft-dark-blue rounded-pill">
                                        4 / 300
                                    </span>
                                </span>
                                                                                                <span role="button" class="badge badge-soft-success rounded-pill" wire:click="addQuestion(135985)">Add</span>
                                                                                            </li>
                                                    </ul>
                        <div class="mt-3">
                            
                        </div>
                    </div>

                    <div class="tab-pane" id="new-list-tab" wire:ignore.self="add-exam-button">
                        <div class="mb-3">
                                                        <label for="exam-name" class="form-label">List Title</label>
                            <input wire:model.defer="examTitle" type="text" placeholder="Enter exam titile" id="exam-title-LIST" class="form-control" required="">
                                                        <div class="text-center">
                                <button class="mt-3 btn btn-primary waves-effect waves-light" wire:key="add-exam-button" wire:click="createExamWithQuestion" wire:loading.attr="disabled">
                                    <span class="btn-label"><i class="fi fi-br-plus"></i></span> Create and Add Question
                                </button>
                            </div>
                                                    </div>
                    </div>
                </div>
            </div>
        </div>

            </div>
    </div>
</div>
    </div>

<!-- Livewire Component wire-end:3QKwM23b9vNoFKuml5LA -->                                <div wire:id="T1rx2Jya2qYaHq5CUJcd">
        <!-- Modal -->
<div x-data="{
        show: window.Livewire.find('T1rx2Jya2qYaHq5CUJcd').entangle('AddToExamsModalOpen').defer,
    }" x-init="() => {

        let el = document.querySelector('#modal-id-d258c9e5f3ceb484dbc4eb6145acf054')

        let modal = new bootstrap.Modal(el);

        $watch('show', value => {
            if (value) {
                modal.show()
            } else {
                modal.hide()
            }
        });

        el.addEventListener('hide.bs.modal', function (event) {
          show = false
        })
    }" wire:ignore.self="" class="modal fade" tabindex="-1" id="modal-id-d258c9e5f3ceb484dbc4eb6145acf054" aria-labelledby="modal-id-d258c9e5f3ceb484dbc4eb6145acf054" aria-hidden="true" x-ref="modal-id-d258c9e5f3ceb484dbc4eb6145acf054">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <h5 class="modal-title"></h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
            <div class="text-center mb-3">
                <span class="fi fi-br-bulb-nib fa-2x mb-2 d-inline-block text-primary"> </span>
                <h4 class="modal-title fs-3 fw-light">                     Add Question to Exams
                    </h4>
            </div>
            <div class="p-2 pt-0">
                <ul class="nav nav-tabs nav-pills1 navtab-bg1 nav-bordered1 justify-content-center show-exam-menu" wire:ignore="add-exam-button">
                    <li class="nav-item ">
                        <a href="#existing-exam-tab" data-bs-toggle="tab" aria-expanded="true" class="buildexam-topnav-item nav-link active nav-white">
                            To Existing Exam
                        </a>
                    </li>
                    <li class="nav-item">
                        <a href="#new-exam-tab" data-bs-toggle="tab" aria-expanded="false" class="buildexam-topnav-item nav-link nav-white">
                            Create New Exam
                        </a>
                    </li>
                </ul>
                <div class="tab-content">
                    <div class="tab-pane show active" id="existing-exam-tab" wire:ignore.self="add-exam-button" wire:key="addToExamsModal">
                        <ul class="list-group">
                                                        <li class="list-group-item d-flex justify-content-between align-items-center py-1">
                                <span class="default-font text-dark fw-bold fs-6">MOCK I
                                    <span role="button" class="badge badge-soft-dark-blue rounded-pill">
                                        2/ 5
                                    </span>
                                </span>
                                                                                                <span role="button" class="badge badge-soft-success rounded-pill" wire:click="addQuestion(135206)">Add</span>
                                                                                            </li>
                                                    </ul>
                        <div class="mt-3">
                            
                        </div>
                    </div>

                    <div class="tab-pane" id="new-exam-tab" wire:ignore.self="add-exam-button">
                        <div class="mb-3">
                                                        <label for="exam-name" class="form-label">Exam Title</label>
                            <input wire:model.defer="examTitle" type="text" placeholder="Enter exam titile" id="exam-title-EXAM" class="form-control" required="">
                                                        <div class="text-center">
                                <button class="mt-3 btn btn-primary waves-effect waves-light" wire:key="add-exam-button" wire:click="createExamWithQuestion" wire:loading.attr="disabled">
                                    <span class="btn-label"><i class="fi fi-br-plus"></i></span> Create and Add Question
                                </button>
                            </div>
                                                    </div>
                    </div>
                </div>
            </div>
        </div>

            </div>
    </div>
</div>
    </div>

<!-- Livewire Component wire-end:T1rx2Jya2qYaHq5CUJcd -->
                            </li>
                                                                                        <li role="button" class="nav-item" style="position:relative;">
                            <a role="button" class="nav-link nav-no-border dropdown-toggle" data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="true">
                                <i class="text-center fi fi-br-paper-plane d-block"></i>
                                <span style="font-size:0.6rem; font-weight:700"> Share</span>
                            </a>
                            <div class="dropdown-menu" data-popper-placement="bottom-start" style="position: absolute; inset: auto auto 0px 0px; margin: 0px; transform: translate3d(0px, -39px, 0px);min-width:300px;">
                                <form class="px-4 py-3">
                                    <div class="mb-2">
                                        <label for="qshare_link" class="form-label">Share Link</label>
                                        <input type="email" class="form-control" id="qshare_link" value="https://www.exam-mate.com/question/179156">
                                    </div>


                                    <button class="copy-link btn btn-primary" data-clipboard-target="#qshare_link">Copy
                                        Link</button>
                                </form>
                            </div>
                        </li>
                                                <li id="prev-btn" role="button" class="nav-item" onclick="javascript:selectPreviousQuestion(179156 ,1)">
                            <a class="nav-link nav-no-border">
                                <i class="text-center fi fi-br-arrow-up d-block"></i>
                                <span style="font-size:0.6rem; font-weight:700"> Previous</span>
                            </a>
                        </li>
                        <li id="next-btn" role="button" class="nav-item" onclick="javascript:selectNextQuestion(179156,1)">
                            <a class="nav-link nav-no-border">
                                <i class="text-center fi fi-br-arrow-down d-block"></i>
                                <span style="font-size:0.6rem; font-weight:700"> Next</span>
                            </a>
                        </li>
                    


                    <li class="nav-item question-tab" wire:ignore="" wire:key="question">
                        <a id="#questions-tab1" href="#questions-tab1" data-bs-toggle="tab" aria-expanded="false" class="nav-link active nav-no-border">
                            <i class="text-center fi fi-br-list d-block"></i>
                            <span style="font-size:0.6rem; font-weight:700"> Question</span>
                        </a>
                    </li>
                                            <li class="nav-item answer-tab" wire:ignore="" wire:key="answer">
                            <a id="#answers-tab1" href="#answers-tab1" data-bs-toggle="tab" aria-expanded="false" class="nav-link nav-no-border">
                                <i class="text-center fi fi-br-key d-block"></i>
                                <span style="font-size:0.6rem; font-weight:700">Answer</span>
                            </a>
                        </li>
                                                                                    </ul>
            </div>
        </div>
        
        <div class="pt-0 overflow-auto bg-white tab-content question-preview" style="min-height: 80vh; max-height:80vh">
            <div class="qpreview-box-loading" wire:loading.delay="">
                <div class="spinner-border text-danger" role="status"></div>
            </div>

            <div class="p-1 d-block q-topics">
                <span wire:ignore="" class="p-1 fs-6 text-red fw-bold " id="topics-title-1">Topic(s):
                    <span id="topics-1" class="text-dark fw-normal fst-italic">Characteristics And Classification Of Living Organisms, Plant Nutrition, Transport In Plants</span></span>
            </div>
            <div id="question-notfound-1" style="display:none" wire:ignore="">
                <div class="px-3">
    <div class="row justify-content-center bg-transparent">
        <div class="col-md-6">
            <div class="card bg-transparent" style="border:none">

                <div class="card-body p-4">

                    <div class="error-ghost text-center">
                        <svg class="ghost" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" width="127.433px" height="132.743px" viewBox="0 0 127.433 132.743" enable-background="new 0 0 127.433 132.743" xml:space="preserve">
                            <path fill="#f79fac" d="M116.223,125.064c1.032-1.183,1.323-2.73,1.391-3.747V54.76c0,0-4.625-34.875-36.125-44.375
                                        s-66,6.625-72.125,44l-0.781,63.219c0.062,4.197,1.105,6.177,1.808,7.006c1.94,1.811,5.408,3.465,10.099-0.6
                                        c7.5-6.5,8.375-10,12.75-6.875s5.875,9.75,13.625,9.25s12.75-9,13.75-9.625s4.375-1.875,7,1.25s5.375,8.25,12.875,7.875
                                        s12.625-8.375,12.625-8.375s2.25-3.875,7.25,0.375s7.625,9.75,14.375,8.125C114.739,126.01,115.412,125.902,116.223,125.064z"></path>
                            <circle fill="#013E51" cx="86.238" cy="57.885" r="6.667"></circle>
                            <circle fill="#013E51" cx="40.072" cy="57.885" r="6.667"></circle>
                            <path fill="#013E51" d="M71.916,62.782c0.05-1.108-0.809-2.046-1.917-2.095c-0.673-0.03-1.28,0.279-1.667,0.771
                                        c-0.758,0.766-2.483,2.235-4.696,2.358c-1.696,0.094-3.438-0.625-5.191-2.137c-0.003-0.003-0.007-0.006-0.011-0.009l0.002,0.005
                                        c-0.332-0.294-0.757-0.488-1.235-0.509c-1.108-0.049-2.046,0.809-2.095,1.917c-0.032,0.724,0.327,1.37,0.887,1.749
                                        c-0.001,0-0.002-0.001-0.003-0.001c2.221,1.871,4.536,2.88,6.912,2.986c0.333,0.014,0.67,0.012,1.007-0.01
                                        c3.163-0.191,5.572-1.942,6.888-3.166l0.452-0.453c0.021-0.019,0.04-0.041,0.06-0.061l0.034-0.034
                                        c-0.007,0.007-0.015,0.014-0.021,0.02C71.666,63.771,71.892,63.307,71.916,62.782z"></path>
                            <circle fill="#FCEFED" stroke="#FEEBE6" stroke-miterlimit="10" cx="18.614" cy="99.426" r="3.292"></circle>
                            <circle fill="#FCEFED" stroke="#FEEBE6" stroke-miterlimit="10" cx="95.364" cy="28.676" r="3.291"></circle>
                            <circle fill="#FCEFED" stroke="#FEEBE6" stroke-miterlimit="10" cx="24.739" cy="93.551" r="2.667"></circle>
                            <circle fill="#FCEFED" stroke="#FEEBE6" stroke-miterlimit="10" cx="101.489" cy="33.051" r="2.666"></circle>
                            <circle fill="#FCEFED" stroke="#FEEBE6" stroke-miterlimit="10" cx="18.738" cy="87.717" r="2.833"></circle>
                            <path fill="#FCEFED" stroke="#FEEBE6" stroke-miterlimit="10" d="M116.279,55.814c-0.021-0.286-2.323-28.744-30.221-41.012
                                        c-7.806-3.433-15.777-5.173-23.691-5.173c-16.889,0-30.283,7.783-37.187,15.067c-9.229,9.736-13.84,26.712-14.191,30.259
                                        l-0.748,62.332c0.149,2.133,1.389,6.167,5.019,6.167c1.891,0,4.074-1.083,6.672-3.311c4.96-4.251,7.424-6.295,9.226-6.295
                                        c1.339,0,2.712,1.213,5.102,3.762c4.121,4.396,7.461,6.355,10.833,6.355c2.713,0,5.311-1.296,7.942-3.962
                                        c3.104-3.145,5.701-5.239,8.285-5.239c2.116,0,4.441,1.421,7.317,4.473c2.638,2.8,5.674,4.219,9.022,4.219
                                        c4.835,0,8.991-2.959,11.27-5.728l0.086-0.104c1.809-2.2,3.237-3.938,5.312-3.938c2.208,0,5.271,1.942,9.359,5.936
                                        c0.54,0.743,3.552,4.674,6.86,4.674c1.37,0,2.559-0.65,3.531-1.932l0.203-0.268L116.279,55.814z M114.281,121.405
                                        c-0.526,0.599-1.096,0.891-1.734,0.891c-2.053,0-4.51-2.82-5.283-3.907l-0.116-0.136c-4.638-4.541-7.975-6.566-10.82-6.566
                                        c-3.021,0-4.884,2.267-6.857,4.667l-0.086,0.104c-1.896,2.307-5.582,4.999-9.725,4.999c-2.775,0-5.322-1.208-7.567-3.59
                                        c-3.325-3.528-6.03-5.102-8.772-5.102c-3.278,0-6.251,2.332-9.708,5.835c-2.236,2.265-4.368,3.366-6.518,3.366
                                        c-2.772,0-5.664-1.765-9.374-5.723c-2.488-2.654-4.29-4.395-6.561-4.395c-2.515,0-5.045,2.077-10.527,6.777
                                        c-2.727,2.337-4.426,2.828-5.37,2.828c-2.662,0-3.017-4.225-3.021-4.225l0.745-62.163c0.332-3.321,4.767-19.625,13.647-28.995
                                        c3.893-4.106,10.387-8.632,18.602-11.504c-0.458,0.503-0.744,1.165-0.744,1.898c0,1.565,1.269,2.833,2.833,2.833
                                        c1.564,0,2.833-1.269,2.833-2.833c0-1.355-0.954-2.485-2.226-2.764c4.419-1.285,9.269-2.074,14.437-2.074
                                        c7.636,0,15.336,1.684,22.887,5.004c26.766,11.771,29.011,39.047,29.027,39.251V121.405z"></path>
                        </svg>
                    </div>

                    <div class="text-center">
                        <h3 class="mt-4">No Questions Found</h3>
                        <p class="text-muted mb-0">we weren't able to find any questions matching your filters. Please try using different filter parameters.</p>
                    </div>

                </div> <!-- end card-body -->
            </div>
            <!-- end card -->
        </div> <!-- end col -->
    </div>
</div>
            </div>
            <div class="tab-pane show active" id="questions-tab1" wire:ignore="">
                <div wire:ignore="">
                    <span class="p-1 fs-5 text-dark fw-bold" id="question-text-1"></span>
                    <img id="" src="https://www.exam-mate.com/storage/water_mark_logo.webp" style="width: 400px;left:0px;position: absolute;top: 100px;display: inline-block;opacity:0.1">                    <div id="question-image-box-1"><div>
                <img class="img-fluid" src="https://www.exam-mate.com/media/questions/474241/0610_w24_qp_43_Q2_Part-1.png" style="max-width: 98%; width: 1000px;">
            </div><div>
                <img class="img-fluid" src="https://www.exam-mate.com/media/questions/474242/0610_w24_qp_43_Q2_Part-2.png" style="max-width: 98%; width: 1000px;">
            </div><div>
                <img class="img-fluid" src="https://www.exam-mate.com/media/questions/474243/0610_w24_qp_43_Q2_Part-3.png" style="max-width: 98%; width: 1000px;">
            </div><div>
                <img class="img-fluid" src="https://www.exam-mate.com/media/questions/474244/0610_w24_qp_43_Q2_Part-4.png" style="max-width: 98%; width: 1000px;">
            </div></div>

                </div>
            </div>
            <div class="tab-pane" id="answers-tab1" wire:ignore="">
                <div wire:ignore="">
                    <span class="p-1 fs-5 text-dark fw-bold" id="answer-text-1"></span>
                    <img id="" src="https://www.exam-mate.com/storage/water_mark_logo.webp" style="width: 400px;left:0px;position: absolute;top: 100px;display: inline-block;opacity:0.1">                    <div id="answer-image-box-1"><div>
                <img class="img-fluid" src="https://www.exam-mate.com/media/questions/474256/0610_w24_ms_43_Q2_1.png" style="max-width: 98%; width: 1000px;">
            </div><div>
                <img class="img-fluid" src="https://www.exam-mate.com/media/questions/474257/0610_w24_ms_43_Q2_2.png" style="max-width: 98%; width: 1000px;">
            </div></div>

                </div>
            </div>
            <div class="tab-pane" id="video-tab1">
                <div class="ratio ratio-16x9">
                    <iframe class="" allowfullscreen="" src="">
                    </iframe>
                </div>
            </div>
            <!--<div class="tab-pane" id="ai-tab1">-->
            <!--    <div class="ratio ratio-16x9">-->
            <!--        <iframe class="" allowfullscreen src="">-->
            <!--        </iframe>-->
            <!--    </div>-->
            <!--</div>-->
        </div>
        <div class="d-none">
            <div id="imageTemplate">
                <img class="img-fluid" src="" style="max-width:98%;">
            </div>
        </div>
    </div>

</div>

<!-- Livewire Component wire-end:TkCPbiaDZKlLcbdI4QuM -->        </div>
    </div>
            </div>
            <!-- content -->

            <!-- Footer Start -->
<footer class="footer">
    <div class="container-fluid">
        <div class="row">
            <div class="col-md-6">
                2025 © exam-mate
            </div>
            <div class="col-md-6">
                <div class="text-md-end footer-links d-none d-sm-block">
                                                                                                                                                            <a href="https://www.exam-mate.com/schools_using_exammate">Schools</a>
                                                                                                                                                                                                 <a href="https://www.exam-mate.com/blog">Blog</a>
                                                                                                                                                                                                 <a href="https://www.exam-mate.com/terms-of-service">T &amp; C</a>
                                                                                        </div>
            </div>
        </div>
    </div>
</footer>
<!-- end Footer -->
        </div>

        <!-- ============================================================== -->
        <!-- End Page content -->
        <!-- ============================================================== -->

    </div>
    <!-- END wrapper -->

    

    
    <script src="/livewire/livewire.js?id=90730a3b0e7144480175" data-turbo-eval="false" data-turbolinks-eval="false"></script><script data-turbo-eval="false" data-turbolinks-eval="false">window.livewire = new Livewire();window.Livewire = window.livewire;window.livewire_app_url = '';window.livewire_token = 'Hl3szbwt5vFLH8eskzLZalI8chsWb3Iw7opK5Ieq';window.deferLoadingAlpine = function (callback) {window.addEventListener('livewire:load', function () {callback();});};let started = false;window.addEventListener('alpine:initializing', function () {if (! started) {window.livewire.start();started = true;}});document.addEventListener("DOMContentLoaded", function () {if (! started) {window.livewire.start();started = true;}});</script>
    
    <!-- bundle -->
<!-- Vendor js -->
<script src="https://www.exam-mate.com/assets/js/vendor.min.js"></script>
<script src="https://www.exam-mate.com/assets/libs/jquery-toast-plugin/jquery-toast-plugin.min.js"></script>
<script src="https://www.exam-mate.com/assets/libs/tippy.js/tippy.js.min.js"></script>
<!-- App js -->
<script src="https://www.exam-mate.com/assets/js/app.js?id=4b7415d7c840d2a1bd37"></script>

<script src="https://www.exam-mate.com/js/main.js?id=b82b59c919ce24214e19"></script>
                  <script>
     window.addEventListener('subjectChanged', event => {
         initZoneBox();
     });

    window.addEventListener('filterResults', event => {
         initZoneBox();
     });


    function initZoneBox() {
        updateZoneBox();
         $(document).ready(function() {
             //set initial state.
             $("input[name='zones[]']").change(function() {
                 updateZoneBox();
             });
         });
     }

     function updateZoneBox(){
        let checked = $("input[name='zones[]']:checked").length;
                 if(checked == 1){
                    let selected = $("input[name='zones[]']:checked").attr('data-name');
                    $('#zones-selected').html(selected);
                 }else if (checked > 1) {
                     $('#zones-selected').html(checked + " Selected");
                 } else {
                     $('#zones-selected').html("Select Zone");
                 }
     }
     
    initZoneBox();
 </script>
  <script>
     window.addEventListener('subjectChanged', event => {
         initSeasonsBox();
     });

     window.addEventListener('filterResults', event => {
         updateSeasonBox();
     });

     function initSeasonsBox() {
         $(document).ready(function() {
            updateSeasonBox();
             //set initial state.
             $("input[name='seasons[]']").change(function() {
                 updateSeasonBox();
             });
         });
     }

     function updateSeasonBox() {
         let checked = $("input[name='seasons[]']:checked").length;
         if (checked == 1) {
             let selected = $("input[name='seasons[]']:checked").attr('data-name');
             $('#seasons-selected').html(selected);
         } else if (checked > 0) {
             $('#seasons-selected').html(checked + " Selected");
         } else {
             $('#seasons-selected').html("Select Season");
         }
     }
     initSeasonsBox();

 </script>
      <script>
         window.addEventListener('subjectChanged', event => {
             initYearsBox();
         });

         window.addEventListener('filterResults', event => {
             initYearsBox();
         });

         function initYearsBox() {
             $(document).ready(function() {
                 updateYearBox();
                 //set initial state.
                 $("input[name='years[]']").change(function() {
                     updateYearBox();
                 });
             });
         }

         function updateYearBox() {
             let numChecked = $("input[name='years[]']:checked").length;
             if (numChecked >= 1 && numChecked <= 3) {
                 let selected = $("input[name='years[]']:checked");
                 let checked = [];
                 selected.each(function() {
                     checked.push($(this).val())
                 });
                 $('#years-selected').html(checked.join(","));
             } else if (numChecked >= 4) {
                 $('#years-selected').html(numChecked + " Selected");
             } else {
                 $('#years-selected').html("Select Year");
             }
         }

         initYearsBox();

     </script>
          <script>
         window.addEventListener('subjectChanged', event => {
             initPapersBox();
         });

         window.addEventListener('filterResults', event => {
             initPapersBox();
         });


         function initPapersBox() {
             $(document).ready(function() {
                 //set initial state.
                 updatePaperBox();
                 $("input[name='papers[]']").change(function() {
                     updatePaperBox();
                 });
             });
         }

         function updatePaperBox() {
             let checked = $("input[name='papers[]']:checked").length;

             if (checked == 1) {
                 let selected = $("input[name='papers[]']:checked").attr('data-name');
                 $('#papers-selected').html(selected);
             } else if (checked > 0) {
                 $('#papers-selected').html(checked + " Selected");
             } else {
                 let name = $("#papers-selected").attr('data-name');
                 $('#papers-selected').html(name);
                 console.log(name);
             }
         }
         initPapersBox();
     </script>
      <script>
         window.addEventListener('subjectChanged', event => {
             initTopicsBox();
         });
         
         window.addEventListener('filterResults', event => {
             initTopicsBox();
         });

         function initTopicsBox() {
             $(document).ready(function() {
                 //set initial state.
                 updateTopicBox();
                 $("input[name='topics[]']").change(function() {
                     updateTopicBox();
                 });

             });
         }

         function updateTopicBox() {
             let checked = $("input[name='topics[]']:checked").length;
             if (checked == 1) {
                 let selected = $("input[name='topics[]']:checked").attr('data-name');
                 if (selected.length > 20) {
                     $('#topics-selected').html(selected.substring(0, 20) + '...');
                 } else {
                     $('#topics-selected').html(selected);
                 }
             } else if (checked > 1) {
                 $('#topics-selected').html(checked + " Selected");
             } else {
                 $('#topics-selected').html("Select Topic");
             }
         }

         initTopicsBox();
     </script>
 
<script>
    $(function() {
        var clipboard = new ClipboardJS('.copy-link');
        clipboard.on('success', function(e) {
            $.toast({
                heading: "Success"
                , text: "Link Copied sucessfully"
                , position: 'bottom-right'
                , icon: 'success'
                , stack: '3'
            });
        });
    });

</script>





<div id="c4g-content-root" class="c4g-widget" data-darkreader-mode="dynamic" data-darkreader-scheme="dark"><meta name="darkreader" content="8cb65737fb6e422eb75ae45bf8b3bc85"></div></body><grammarly-desktop-integration data-grammarly-shadow-root="true"></grammarly-desktop-integration></html>
"""

# Extract all question_image URLs
question_links = re.findall(
    r'"question_images":\["(.*?)"\]', 
    html_content.replace("&quot;", '"')
)

# Flatten and dedupe URLs
question_links = list(set(
    url for group in question_links 
    for url in group.split('","')
))

# Download files
os.makedirs("question_papers", exist_ok=True)
for url in question_links:
    try:
        response = requests.get(url)
        filename = os.path.join("question_papers", url.split("/")[-1])
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed {url}: {e}")