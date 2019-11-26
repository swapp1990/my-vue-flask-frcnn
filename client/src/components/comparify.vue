    <!-- <div class="handle">
        <svg class="handle__arrow handle__arrow--l" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
        <svg class="handle__arrow handle__arrow--r" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
    </div> -->
<template>
    <div class="compare-wrapper">
        <div class="compare">
            <div class="compare__content" :style="{'width': width}">
                <slot name="first"></slot>
            </div>
            <resize-observer @notify="handleResize"></resize-observer>
            <div class="handle-wrap" :style="{left:`calc(${compareWidth + '%'} - 2px / 2`}">

                <span class="handle-line"></span>
            </div>
            <div class="compare-overlay " :style="{width:`calc(${compareWidth + '%'})`}">
                <div class="compare-overlay__content" :style="{ 'width': width}">
                    <slot name="second"></slot>
                </div>
            </div>
                
            </div>
        </div>
    </div>
</template>
<script>
export default {
    name: "comparify",
    props: {
        value: {default: 50}
    },
    data() {
        return {
            width: null,
            compareWidth: this.value
        }
    },
    watch:{
        value(){
            this.compareWidth= this.value
        }
    },
    mounted(){
        this.width = this.getContainerWidth();
    },
    methods:{
        handleInput(e){
            this.compareWidth = e.target.value;
            this.$emit('input', e.target.value);
        },
        handleResize(){
            const w = this.getContainerWidth();
            if(w === this.width)
                return;
            this.width = w;
            console.log(this.width);
        },
        getContainerWidth(){
            return window.getComputedStyle(this.$el,null).getPropertyValue('width')
        }
    }
}
</script>
<style lang="scss" scoped>
    :root{
    --handle-bg: blue;
    --handle-width: 30px;
    --handle-height: 30px;
    --handle-chevron-size: 20px;
    
    --handle-line-bg: blue;
    --handle-line-width: 2px;
    --handle-line-height: 100%;
    
    --z-index-handle: 5;
    --z-index-handle-line: 4;
    --z-index-range-input: 6;
    }
    .compare-wrapper{
        position: relative;
    }
    .compare, .compare__content {
        position: relative;
        height: 100%;
    }
    .compare-overlay{
        position: absolute;
        overflow:hidden;
        height: 100%;
        top:0;
    }
.compare-overlay__content{
  position:relative;
  height: 100%;
  width: 100%;
}
    .handle__arrow{
        position: absolute;
        width: 20px;
    }
    .handle__arrow--l{
        left:0;
    }
    .handle__arrow--r{
        right:0;
    }
    .handle-wrap{
        display: flex;
        align-items: center;
        justify-content: center;
        position: absolute;
        top: 50%;
        height: 100%;
        transform: translate(-50%, -50%);
        z-index: 5;
    }
    .handle{
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        background: blue;
        border-radius: 50%;
        width: 30px;
        height: 30px;
    }
    .handle-line{
        content: '';
        position: absolute;
        top:0;
        width: 2px;
        height: 100%;
        background: blue;
        z-index: 4;
        pointer-events:none;
        user-select:none;
    }
</style>

// Vue.component('comparify', {
//   props:{
//     value: { default: 50 },
//     step: { default: '.1' }
//   },
//   template:`#compare-template`,
//   data(){
//     return {
//       width: null,
//       compareWidth: this.value,
//     }
//   },
//   watch:{
//     value(){
//       this.compareWidth= this.value
//     }
//   },
//   mounted(){
//     this.width = this.getContainerWidth();
//   },
//   methods:{
//     handleInput(e){
//       this.compareWidth = e.target.value
//       this.$emit('input', e.target.value)
//     },
//     handleResize(){
//       const w = this.getContainerWidth();
//       if(w === this.width)
//         return;
//       this.width = w
//       console.log(this.width)
//     },
//     getContainerWidth(){
//       return window.getComputedStyle(this.$el,null).getPropertyValue('width')
//     },
//   }
// })