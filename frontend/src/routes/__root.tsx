import { createRootRoute, Link, Outlet } from '@tanstack/react-router'
import { NavigationMenu, 
  NavigationMenuLink,
  NavigationMenuItem, 
  NavigationMenuList, 
  navigationMenuTriggerStyle 
} from "@/components/ui/navigation-menu"


function RootLayout() {
  return (
    <div className="mx-auto w-full h-screen container">
      {/* Navigation */}
      <div className='place-items-center grid'>
        <NavigationMenu className="z-5 mt-2 md:mt-4 w-full sm:w-auto max-w-full sm:max-w-auto">
          <NavigationMenuList>
            <NavigationMenuItem>
              <Link to='/'>
                <NavigationMenuLink className={navigationMenuTriggerStyle()}>
                  Home
                </NavigationMenuLink>
              </Link>
            </NavigationMenuItem>

            <NavigationMenuItem>
              <Link to='/blog'>
                <NavigationMenuLink className={navigationMenuTriggerStyle()}>
                  How did I do it?
                </NavigationMenuLink>
              </Link>
            </NavigationMenuItem>

            <NavigationMenuItem>
              <a
                href='https://github.com/chheangg/dog_and_cat_classifier_cnn_from_scratch' 
                target='_blank'
                rel="noopener noreferrer"
              >
                <NavigationMenuLink className={navigationMenuTriggerStyle()}>
                  Github
                </NavigationMenuLink>
              </a>
            </NavigationMenuItem>
          </NavigationMenuList>
        </NavigationMenu>
      </div>
      <main className='mt-8 lg:mt-8 xl:mt-12'>
        <Outlet />
      </main>
    </div>
  )
}

export const Route = createRootRoute({ component: RootLayout })