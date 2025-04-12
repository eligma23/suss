PATH_EXCLUSIONS = [
    # Byte-compiled / optimized / DLL files
    "(^|/)__pycache__/",
    # Distribution / packaging
    "(^|/)build/",
    "(^|/)dist/",
    "(^|/)develop-eggs/",
    "(^|/)downloads/",
    "(^|/)eggs/",
    "(^|/)\.eggs/",
    "(^|/)lib/",
    "(^|/)lib64/",
    "(^|/)parts/",
    "(^|/)sdist/",
    "(^|/)var/",
    "(^|/)wheels/",
    "(^|/)pip-wheel-metadata/",
    "(^|/)share/python-wheels/",
    "(^|/)\.egg-info/",
    # Unit test / coverage reports
    "(^|/)htmlcov/",
    "(^|/)\.tox/",
    "(^|/)\.nox/",
    "(^|/)\.hypothesis/",
    "(^|/)\.pytest_cache/",
    # PyBuilder
    "(^|/)target/",
    # IPython
    "(^|/)profile_default/",
    "(^|/)ipython_config\.py$",
    # PEP 582; used by e.g. github.com/David-OConnor/pyflow
    "(^|/)__pypackages__/",
    # MyPy
    "(^|/)\.mypy_cache/",
    # Pyre type checker
    "(^|/)\.pyre/",
    # Caches
    "(^|/)cache/",
    # Dependencies
    "^[Dd]ependencies/",
    # Distributions
    # C deps
    "^deps/",
    "(^|/)configure$",
    # .NET Core Install Scripts
    "(^|/)dotnet-install\.(ps1|sh)$",
    # Linters
    "(^|/)cpplint\.py",
    # Node dependencies
    "(^|/)node_modules/",
    # Next
    "(^|/)\.next/",
    "(^|/)out/",
    "(^|/)dist/",
    "(^|/)cache/",
    # Yarn 2
    "(^|/)\.yarn/releases/",
    "(^|/)\.yarn/plugins/",
    "(^|/)\.yarn/sdks/",
    "(^|/)\.yarn/versions/",
    "(^|/)\.yarn/unplugged/",
    # esy.sh dependencies
    "(^|/)_esy$",
    # Bower Components
    "(^|/)bower_components/",
    # Erlang bundles
    "^rebar$",
    # Go dependencies
    "(^|/)Godeps/_workspace/",
    # Go fixtures
    "(^|/)testdata/",
    # Bootstrap css and js
    "(^|/)bootstrap([^/.]*)(?=\.).*\.(js|css|less|scss|styl)$",
    "(^|/)custom\.bootstrap([^\s]*)(js|css|less|scss|styl)$",
    # Select2
    "(^|/)select2/.*\.(css|scss|js)$",
    # Bulma css
    "(^|/)bulma\.(css|sass|scss)$",
    # Vendored dependencies
    "(3rd|[Tt]hird)[-_]?[Pp]arty/",
    "(^|/)vendors?/",
    "(^|/)[Ee]xtern(als?)?/",
    "(^|/)[Vv]+endor/",
    # Debian packaging
    "^debian/",
    # Haxelib projects often contain a neko bytecode file named run.n
    "(^|/)run\.n$",
    # Bootstrap Datepicker
    "(^|/)bootstrap-datepicker/",
    ## Commonly Bundled JavaScript frameworks ##
    # jQuery
    "(^|/)jquery([^.]*)\.js$",
    "(^|/)jquery\-\d\.\d+(\.\d+)?\.js$",
    # jQuery UI
    "(^|/)jquery\-ui(\-\d\.\d+(\.\d+)?)?(\.\w+)?\.(js|css)$",
    "(^|/)jquery\.(ui|effects)\.([^.]*)\.(js|css)$",
    # jQuery Gantt
    "(^|/)jquery\.fn\.gantt\.js",
    # jQuery fancyBox
    "(^|/)jquery\.fancybox\.(js|css)",
    # Fuel UX
    "(^|/)fuelux\.js",
    # jQuery File Upload
    "(^|/)jquery\.fileupload(-\w+)?\.js$",
    # jQuery dataTables
    "(^|/)jquery\.dataTables\.js",
    # bootboxjs
    "(^|/)bootbox\.js",
    # pdf-worker
    "(^|/)pdf\.worker\.js",
    # Slick
    "(^|/)slick\.\w+.js$",
    # Leaflet plugins
    "(^|/)Leaflet\.Coordinates-\d+\.\d+\.\d+\.src\.js$",
    "(^|/)leaflet\.draw-src\.js",
    "(^|/)leaflet\.draw\.css",
    "(^|/)Control\.FullScreen\.css",
    "(^|/)Control\.FullScreen\.js",
    "(^|/)leaflet\.spin\.js",
    "(^|/)wicket-leaflet\.js",
    # Sublime Text workspace files
    "(^|/)\.sublime-project",
    "(^|/)\.sublime-workspace",
    # VS Code workspace files
    "(^|/)\.vscode/",
    # Prototype
    "(^|/)prototype(.*)\.js$",
    "(^|/)effects\.js$",
    "(^|/)controls\.js$",
    "(^|/)dragdrop\.js$",
    # Typescript definition files
    "(.*?)\.d\.ts$",
    # MooTools
    "(^|/)mootools([^.]*)\d+\.\d+.\d+([^.]*)\.js$",
    # Dojo
    "(^|/)dojo\.js$",
    # MochiKit
    "(^|/)MochiKit\.js$",
    # YUI
    "(^|/)yahoo-([^.]*)\.js$",
    "(^|/)yui([^.]*)\.js$",
    # WYS editors
    "(^|/)ckeditor\.js$",
    "(^|/)tiny_mce([^.]*)\.js$",
    "(^|/)tiny_mce/(langs|plugins|themes|utils)",
    # Ace Editor
    "(^|/)ace-builds/",
    # Fontello CSS files
    "(^|/)fontello(.*?)\.css$",
    # MathJax
    "(^|/)MathJax/",
    # Chart.js
    "(^|/)Chart\.js$",
    # CodeMirror
    "(^|/)[Cc]ode[Mm]irror/(\d+\.\d+/)?(lib|mode|theme|addon|keymap|demo)",
    # SyntaxHighlighter - http://alexgorbatchev.com/
    "(^|/)shBrush([^.]*)\.js$",
    "(^|/)shCore\.js$",
    "(^|/)shLegacy\.js$",
    # AngularJS
    "(^|/)angular([^.]*)\.js$",
    # D3.js
    "(^|\/)d3(\.v\d+)?([^.]*)\.js$",
    # React
    "(^|/)react(-[^.]*)?\.js$",
    # flow-typed
    "(^|/)flow-typed/.*\.js$",
    # Modernizr
    "(^|/)modernizr\-\d\.\d+(\.\d+)?\.js$",
    "(^|/)modernizr\.custom\.\d+\.js$",
    # Knockout
    "(^|/)knockout-(\d+\.){3}(debug\.)?js$",
    ## Python ##
    # Sphinx
    "(^|/)docs?/_?(build|themes?|templates?|static)/",
    # django
    "(^|/)admin_media/",
    "(^|/)local_settings\.py",
    # Flask
    "(^|/)instance/",
    # Fabric
    "(^|/)fabfile\.py$",
    # WAF
    "(^|/)waf$",
    # .osx
    "(^|/)\.osx$",
    ## Obj-C ##
    # Xcode
    ### these can be part of a directory name
    "\.xctemplate/",
    "\.imageset/",
    # Carthage
    "(^|/)Carthage/",
    # Sparkle
    "(^|/)Sparkle/",
    # Crashlytics
    "(^|/)Crashlytics\.framework/",
    # Fabric
    "(^|/)Fabric\.framework/",
    # BuddyBuild
    "(^|/)BuddyBuildSDK\.framework/",
    # Realm
    "(^|/)Realm\.framework",
    # RealmSwift
    "(^|/)RealmSwift\.framework",
    # git config files
    "(^|/)\.gitattributes$",
    "(^|/)\.gitignore$",
    "(^|/)\.gitmodules$",
    ## Groovy ##
    # Gradle
    "(^|/)gradlew$",
    "(^|/)gradlew\.bat$",
    "(^|/)gradle/wrapper/",
    ## Java ##
    # Maven
    "(^|/)mvnw$",
    "(^|/)mvnw\.cmd$",
    "(^|/)\.mvn/wrapper/",
    ## .NET ##
    # Visual Studio IntelliSense
    "-vsdoc\.js$",
    "\.intellisense\.js$",
    # jQuery validation plugin (MS bundles this with asp.net mvc)
    "(^|/)jquery([^.]*)\.validate(\.unobtrusive)?\.js$",
    "(^|/)jquery([^.]*)\.unobtrusive\-ajax\.js$",
    # Microsoft Ajax
    "(^|/)[Mm]icrosoft([Mm]vc)?([Aa]jax|[Vv]alidation)(\.debug)?\.js$",
    # NuGet
    "(^|/)[Pp]ackages\/.+\.\d+\/",
    # ExtJS
    "(^|/)extjs/.*?\.js$",
    "(^|/)extjs/.*?\.xml$",
    "(^|/)extjs/.*?\.txt$",
    "(^|/)extjs/.*?\.html$",
    "(^|/)extjs/.*?\.properties$",
    "(^|/)extjs/\.sencha/",
    "(^|/)extjs/docs/",
    "(^|/)extjs/builds/",
    "(^|/)extjs/cmd/",
    "(^|/)extjs/examples/",
    "(^|/)extjs/locale/",
    "(^|/)extjs/packages/",
    "(^|/)extjs/plugins/",
    "(^|/)extjs/resources/",
    "(^|/)extjs/src/",
    "(^|/)extjs/welcome/",
    # Html5shiv
    "(^|/)html5shiv\.js$",
    # Test fixtures
    "(^|/)[Tt]ests?/fixtures/",
    "(^|/)[Ss]pecs?/fixtures/",
    # PhoneGap/Cordova
    "(^|/)cordova([^.]*)\.js$",
    "(^|/)cordova\-\d\.\d(\.\d)?\.js$",
    # Foundation js
    "(^|/)foundation(\..*)?\.js$",
    # Vagrant
    "(^|/)Vagrantfile$",
    # R packages
    "(^|/)vignettes/",
    "(^|/)inst/extdata/",
    # Typesafe Activator
    "(^|/)activator$",
    # - (^|/)activator\.bat$
    # PuPHPet
    "(^|/)puphpet/",
    # Android Google APIs
    "(^|/)\.google_apis/",
    # Jenkins Pipeline
    "(^|/)Jenkinsfile$",
    # GitHub.com
    "(^|/)\.github/",
    # Environments
    "(^|/)venv/",
    "(^|/)env/",
    "(^|/)ENV/",
    "(^|/)env\.bak/",
    "(^|/)venv\.bak/",
    # Lock Files (NPM, Yarn, PNPM, etc.)
    "(^|/)package-lock\.json",
    "(^|/)yarn\.lock",
    "(^|/)pnpm-lock\.ya?ml",
    "(^|/)pnpm-workspace\.ya?ml",
    "(^|/)pnpm-workspace\.json",
    # Lock Files (Python)
    "(^|/)Pipfile\.lock",
    # Lock Files Generic
    "(^|/).*\.lock$",
]
