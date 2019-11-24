<template>
    <div>
        <svg xmlns="http://www.w3.org/2000/svg">
            <symbol id="ellipsis-v" viewBox="0 0 64 512"><path d="M32 224c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zM0 136c0 17.7 14.3 32 32 32s32-14.3 32-32-14.3-32-32-32-32 14.3-32 32zm0 240c0 17.7 14.3 32 32 32s32-14.3 32-32-14.3-32-32-32-32 14.3-32 32z"/></symbol>
            <symbol id="camera" viewBox="0 0 512 512"><path d="M324.3 64c3.3 0 6.3 2.1 7.5 5.2l22.1 58.8H464c8.8 0 16 7.2 16 16v288c0 8.8-7.2 16-16 16H48c-8.8 0-16-7.2-16-16V144c0-8.8 7.2-16 16-16h110.2l20.1-53.6c2.3-6.2 8.3-10.4 15-10.4h131m0-32h-131c-20 0-37.9 12.4-44.9 31.1L136 96H48c-26.5 0-48 21.5-48 48v288c0 26.5 21.5 48 48 48h416c26.5 0 48-21.5 48-48V144c0-26.5-21.5-48-48-48h-88l-14.3-38c-5.8-15.7-20.7-26-37.4-26zM256 408c-66.2 0-120-53.8-120-120s53.8-120 120-120 120 53.8 120 120-53.8 120-120 120zm0-208c-48.5 0-88 39.5-88 88s39.5 88 88 88 88-39.5 88-88-39.5-88-88-88z"/></symbol>
        </svg>
        <div class="header">
            <div>
				<svg class="svg svg--camera" width="24px" height="24px" viewBox="0 0 512 512">
					<use xlink:href="#camera"></use>
				</svg>
			</div>
        </div>
        <stories></stories>
        <posts></posts>
    </div>
</template>
<script>
import Stories from '@/components/Stories.vue';
import Posts from '@/components/Posts.vue';
export default {
    name:'InstaView',
    components: {
        Stories: Stories,
        Posts: Posts
    }
}
</script>
<style lang="scss" scoped>
$base-spacing-unit: 24px;
$half-spacing-unit: $base-spacing-unit / 2;
.header {
	display:flex;
	justify-content:space-between;
	align-items:center;
	background:#FAFAFA;
	padding:$half-spacing-unit;
	border-bottom: 1px solid #eee;
	font-size:0;
}
.svg {
	&--camera {
		max-width:$base-spacing-unit;
		max-height:$base-spacing-unit;
	}
}
</style>