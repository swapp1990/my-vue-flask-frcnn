<template>
    <div>
        <div class="stories">
            <div class="story" v-for="story in stories">
                <img class="story-userimage" :src="story.image">
                <p class="story-username" v-html="story.name"></p>
            </div>
        </div>
    </div>
</template>
<script>
export default {
    name:'Stories',
    data: function () {
		return {
			stories: [
				{name: "User 1", image:"https://via.placeholder.com/75"},
				{name: "User 2", image:"https://via.placeholder.com/75"},
				{name: "User 3", image:"https://via.placeholder.com/75"},
				{name: "User 4", image:"https://via.placeholder.com/75"},
				{name: "User 5", image:"https://via.placeholder.com/75"},
				{name: "User 6", image:"https://via.placeholder.com/75"},
				{name: "User 7", image:"https://via.placeholder.com/75"},
			],
        }
    }
}
</script>
<style lang="scss" scoped>
$base-spacing-unit: 24px;
$half-spacing-unit: $base-spacing-unit/2;
.stories {
    border-bottom: 1px solid #f1f1f1;
    overflow-x: scroll;
    display: flex;
    padding: $half-spacing-unit;

    .story {
        flex: 0 0 auto;
        margin-right: $base-spacing-unit;
        text-align: center;
        &-userimage {
            border-radius: 50%;
            position: relative;
            border:2px solid red;
        }
    }
}
</style>