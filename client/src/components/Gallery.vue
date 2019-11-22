<template>
  <div class="gallery">
    <div class="gallery-panel"
         v-for="photo in activeThreads"
         :key="photo.id">
      <!-- <router-link :to="`/photo/${photo.id}`">
        <img :src="thumbUrl(photo.filename)">
      </router-link> -->
      <div>Test</div>
    </div>
  </div>
</template>

<script>

export default {
  name: 'Gallery',
  props: ['threads_p'],
  watch: {
      activeThreads: function (n, o) {
          console.log(n);
      }
  },
  data() {
    return {
       activeThreads: this.threads_p
    };
  },
  methods: {
    thumbUrl(filename) {
      return require(`../assets/images/thumbnails/${filename}`);
    },
  },
};
</script>

<style>
  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(20rem, 1fr));
    grid-gap: 1rem;
    max-width: 80rem;
    margin: 5rem auto;
    padding: 0 5rem;
  }
  .gallery-panel {
    background-color: aqua;
    width: 100%;
    height: 22vw;
    object-fit: cover;
    border-radius: 0.75rem;
  }
</style>