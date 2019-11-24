<template>
    <div class="posts">
        <div class="post" v-for="post in posts">
            <div class="post__header">
                <div class="post-header">
                    <div class="post-header-user">
                        <div>
                            <img class="post-header-user__image" src="https://via.placeholder.com/50">
                        </div>
                        <div>
                            <p class="post-header-user__name" v-html="post.user.name"></p>
                            <p class="post-header-user__location" v-html="post.location"></p>
                        </div>
                    </div>
                    <div class="post-header-settings">
                        <svg width="24px" height="24px" class="svg svg--settings" viewBox="0 0 512 512">
                            <use xlink:href="#ellipsis-v"></use>
                        </svg>
                    </div>
                </div>
            </div>
            <div class="post__image">
                <img :src="post.image">
            </div>
            <div class="post__likes">
                <p>{{ post.likes }} likes</p>
            </div>
            <div class="post__name" v-html="post.user.name"></div>
            <div class="post__status" v-html="post.status"></div>
            <div class="post__time" v-html="post.time"></div>
        </div>
    </div>
</template>
<script>

export default {
    name:'Posts',
    data: function () {
		return {
			posts: [
            {
                user: { 
                    name: "Name 1",
                    image: "https://via.placeholder.com/75"
                },
                location: "Middlesbrough, UK",
                status: "Example Status",
                time: "2 Days Ago",
                comments: [
                    {name: "Name", text: "Comment"}
                ],
                likes: 4300,
                image: "https://images.unsplash.com/photo-1493235431945-90c060301e41?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=0279978342b1a6b4f3e0e00a0784c2e3&dpr=1&auto=format&fit=crop&w=1000&q=80&cs=tinysrgb"
            },
            {
                user: { 
                    name: "Name 2",
                    image: "https://via.placeholder.com/75"
                },
                location: "Middlesbrough, UK",
                status: "Example Status",
                time: "3 Days Ago",
                comments: [
                    {name: "Name", text: "Comment"}
                ],
                likes: 4300,
                image: "https://images.unsplash.com/photo-1522202222206-b75023c48f4f?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=9382691bc04f781d30f6c9cb3738e91b&dpr=1&auto=format&fit=crop&w=1000&q=80&cs=tinysrgb"
            },
            {
                user: { 
                    name: "Name 3",
                    image: "https://via.placeholder.com/75"
                },
                location: "Middlesbrough, UK",
                status: "Example Status",
                time: "3 Days Ago",
                comments: [
                    {name: "Name", text: "Comment"}
                ],
                likes: 4300,
                image: "https://images.unsplash.com/photo-1522202222206-b75023c48f4f?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=9382691bc04f781d30f6c9cb3738e91b&dpr=1&auto=format&fit=crop&w=1000&q=80&cs=tinysrgb"
            },
            {
                user: { 
                    name: "Name 3",
                    image: "https://via.placeholder.com/75"
                },
                location: "Middlesbrough, UK",
                status: "Example Status",
                time: "3 Days Ago",
                comments: [
                    {name: "Name", text: "Comment"}
                ],
                likes: 4300,
                image: "https://images.unsplash.com/photo-1522202222206-b75023c48f4f?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=9382691bc04f781d30f6c9cb3738e91b&dpr=1&auto=format&fit=crop&w=1000&q=80&cs=tinysrgb"
            },
            {
                user: { 
                    name: "Name 3",
                    image: "https://via.placeholder.com/75"
                },
                location: "Middlesbrough, UK",
                status: "Example Status",
                time: "3 Days Ago",
                comments: [
                    {name: "Name", text: "Comment"}
                ],
                likes: 4300,
                image: "https://images.unsplash.com/photo-1522202222206-b75023c48f4f?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=9382691bc04f781d30f6c9cb3738e91b&dpr=1&auto=format&fit=crop&w=1000&q=80&cs=tinysrgb"
            },
            {
                user: { 
                    name: "Name 3",
                    image: "https://via.placeholder.com/75"
                },
                location: "Middlesbrough, UK",
                status: "Example Status",
                time: "3 Days Ago",
                comments: [
                    {name: "Name", text: "Comment"}
                ],
                likes: 4300,
                image: "https://images.unsplash.com/photo-1522202222206-b75023c48f4f?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=9382691bc04f781d30f6c9cb3738e91b&dpr=1&auto=format&fit=crop&w=1000&q=80&cs=tinysrgb"
            }
            ]
        }
    },
}   
</script>
<style lang="scss"  scoped>
$base-spacing-unit: 24px;
$half-spacing-unit: $base-spacing-unit/2;
.svg {
    &--settings {
		fill: #999999;
		max-width:$base-spacing-unit;
	}
}
.posts {
    padding-bottom: $base-spacing-unit*3;
    display: grid;
    grid-template-columns: repeat(3,1fr);
}
//Single post
.post {
    &__header {
        padding:$half-spacing-unit;
    }
    &-header {
        display:flex;
        align-items:center;
        &-user {
            flex: 1 0 75%;
            display: flex;
            align-items:center;
            &__image {
                border-radius: 50%;
                border:2px solid red;
                margin-right: $half-spacing-unit;
            }
            &__name {
                font-weight: bold;
            }
        }
    }
    &__image {
        img {
            width: 100%;
            max-width: 100%;
            height: auto;
        }
    }
    &__likes {
		margin-top:$half-spacing-unit;
		margin-bottom:$half-spacing-unit / 4;
		font-weight:bold;
	}
    &__status {
		display:inline-block;
		margin-bottom: $half-spacing-unit / 4;
	}
    &__name {
		font-weight:bold;
		display:inline-block;
	}
    &__time {
		font-weight:lighter;
		color:#787878;
	}
}

</style>