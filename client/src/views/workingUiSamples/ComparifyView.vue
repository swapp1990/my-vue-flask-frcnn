<template>
    <!-- <div class="h-screen bg-grey-light font-sans overflow-auto px-2 py-8">
        <div class="my-8 p-4 max-w-lg mx-auto bg-white shadow border w-full rounded">
            <h3 class="text-xl mb-4 text-grey-darkest">Compare Elements</h3>
            <comparify class="bg-white h-64 w-full">
                <img slot="first" class="h-64 object-fit-cover h-full w-full" src="https://images.unsplash.com/photo-1525936607120-6a787fb05d6a?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=3cb09a29812d04718a5855dc61442612&auto=format&fit=crop&w=800&q=90" alt="">
                <img slot="second" class="object-fit-cover h-full w-full" src="https://images.unsplash.com/photo-1510149053388-87292573d661?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=18d7ef1ac6ffcd0a644aaa1d330278c6&auto=format&fit=crop&w=800&q=90" alt="">
            </comparify>
        </div>
    </div> -->
    <div class="demo">
        <h1>Hello world!</h1>
        <div class="resized">
        <textarea :style="{ width: `${width}px` }"></textarea>
        <resize-observer @notify="handleResize"></resize-observer>
        </div>
    </div>
</template>
<script>
import comparify from '@/components/comparify.vue';
export default {
    name: 'ComparifyView',
    components: {
        comparify: comparify
    },
    data () {
        return {
            count: 0,
            show: true,
            width: 300,
        }
    },
    methods: {
        handleResize () {
        console.log('resized')
        }
    }
}
</script>
<style scoped>
.resized {
  position: relative;
  border: solid 1px #42b983;
  margin: 12px;
  padding: 12px;
  display: inline-block;
}
.demo {
  position: relative;
}
</style>